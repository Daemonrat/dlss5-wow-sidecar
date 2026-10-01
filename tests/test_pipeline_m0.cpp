#include <catch2/catch_test_macros.hpp>
#include <windows.h>
#include <chrono>
#include <atomic>
#include <memory>
#include <thread>

#include "core/GpuProfile.h"
#include "neural/PassthroughPass.h"
#include "runtime/Pipeline.h"

using namespace sidecar;
using namespace std::chrono_literals;

namespace {

struct PatternApp {
  PROCESS_INFORMATION pi{};
  HWND hwnd = nullptr;
  bool Launch() {
    STARTUPINFOW si{sizeof(si)};
    wchar_t cmd[] = L"testpattern.exe";
    if (!CreateProcessW(nullptr, cmd, nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &si, &pi)) return false;
    for (int i = 0; i < 200 && !hwnd; ++i) {
      hwnd = FindWindowW(L"SidecarTestPattern", nullptr);
      if (!hwnd) std::this_thread::sleep_for(25ms);
    }
    return hwnd != nullptr;
  }
  ~PatternApp() {
    if (pi.hProcess) {
      TerminateProcess(pi.hProcess, 0);
      CloseHandle(pi.hProcess);
      CloseHandle(pi.hThread);
    }
  }
};

// The overlay window is owned by this thread, so window state the render
// thread requests is only applied while this thread pumps. wWinMain runs a
// GetMessage loop for exactly this reason; the test has to do the same.
void PumpFor(std::chrono::milliseconds duration) {
  const auto until = std::chrono::steady_clock::now() + duration;
  MSG msg{};
  while (std::chrono::steady_clock::now() < until) {
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    std::this_thread::sleep_for(2ms);
  }
}

}  // namespace

TEST_CASE("background pause preserves manual off and resumes processing", "[device]") {
  struct CountingPass : INeuralPass {
    std::atomic<unsigned> frames{0}, resets{0};
    std::unique_ptr<INeuralPass> copy = PassthroughPass::Create();
    bool Evaluate(ID3D12GraphicsCommandList* cl, ID3D12Resource* color,
                  ID3D12Resource* motion, ID3D12Resource* depth,
                  ID3D12Resource* out) override {
      ++frames;
      return copy->Evaluate(cl, color, motion, depth, out);
    }
    void ResetHistory() override { ++resets; }
    const char* Name() const override { return "counting passthrough"; }
  };
  auto gpu = DetectPrimaryGpu();
  REQUIRE(gpu.has_value());
  PatternApp app;
  REQUIRE(app.Launch());
  PipelineConfig cfg;
  cfg.target = app.hwnd;
  auto pass = std::make_unique<CountingPass>();
  auto* counter = pass.get();
  auto pipeline = Pipeline::Create(*gpu, cfg, std::move(pass));
  REQUIRE(pipeline);
  pipeline->Start();
  PumpFor(1500ms);
  REQUIRE(counter->frames.load() > 0);
  pipeline->SetTargetForeground(false);
  PumpFor(300ms);
  CHECK_FALSE(IsWindowVisible(pipeline->OverlayHwnd()));
  if (pipeline->GetHud()) CHECK_FALSE(IsWindowVisible(pipeline->GetHud()->Hwnd()));
  const auto pausedFrames = counter->frames.load();
  PumpFor(300ms);
  CHECK(counter->frames.load() == pausedFrames);
  pipeline->SetOverlayVisible(false);
  pipeline->SetTargetForeground(true);
  PumpFor(300ms);
  CHECK_FALSE(IsWindowVisible(pipeline->OverlayHwnd()));
  CHECK(counter->frames.load() == pausedFrames);
  pipeline->SetOverlayVisible(true);
  PumpFor(1000ms);
  CHECK(IsWindowVisible(pipeline->OverlayHwnd()));
  CHECK(counter->frames.load() > pausedFrames);
  CHECK(counter->resets.load() > 0);
  pipeline->Panic();
  PumpFor(100ms);
  pipeline->SetOverlayVisible(true);
  pipeline->SetTargetForeground(true);
  CHECK_FALSE(IsWindowVisible(pipeline->OverlayHwnd()));
  pipeline->Stop();
}

TEST_CASE("pipeline runs end to end over the test pattern", "[device]") {
  auto gpu = DetectPrimaryGpu();
  REQUIRE(gpu.has_value());

  PatternApp app;
  REQUIRE(app.Launch());

  PipelineConfig cfg;
  cfg.target = app.hwnd;
  cfg.showOverlay = true;

  auto pipeline = Pipeline::Create(*gpu, cfg, PassthroughPass::Create());
  REQUIRE(pipeline != nullptr);
  pipeline->Start();

  // Let it settle, then require real throughput.
  std::this_thread::sleep_for(3s);
  REQUIRE(pipeline->Running());

  const auto& stats = pipeline->Stats();
  INFO("p50=" << stats.P50() << "ms p99=" << stats.P99()
              << "ms drops=" << stats.Dropped());
  REQUIRE(stats.Count() >= 30);
  REQUIRE(stats.P50() > 0.0);

  // Sanity bound: anything above this means the seam is fundamentally broken,
  // not merely slow. The real M1 judgement is made against live WoW by a human
  // reading the HUD, not by this assertion.
  REQUIRE(stats.P99() < 250.0);

  pipeline->Stop();
  REQUIRE(pipeline->Running() == false);
}

// M3 Task 8, step 4. The masked path adds a second render target and four more
// barrier transitions per frame, and a wrong resource state is the kind of bug
// that shows up as a device removal under load rather than a wrong pixel. So
// this runs the whole pipeline with a mask configured and requires it to keep
// presenting at the same throughput as the unmasked path.
//
// It cannot assert what the blend produced -- with PassthroughPass the neural
// input and the original are the same image, so the blend is an identity by
// construction. UiMask's own [device] tests cover the pixels; this covers the
// plumbing.
TEST_CASE("pipeline runs with a UI mask configured", "[device]") {
  auto gpu = DetectPrimaryGpu();
  REQUIRE(gpu.has_value());

  PatternApp app;
  REQUIRE(app.Launch());

  PipelineConfig cfg;
  cfg.target = app.hwnd;
  cfg.showOverlay = true;
  // Two overlapping rectangles, so the union path and the feather both run.
  cfg.uiMaskRects = {UiRect{0, 0, 200, 120}, UiRect{150, 80, 400, 300}};
  cfg.uiMaskFeather = 6;

  auto pipeline = Pipeline::Create(*gpu, cfg, PassthroughPass::Create());
  REQUIRE(pipeline != nullptr);
  pipeline->Start();

  std::this_thread::sleep_for(3s);
  INFO("lastError='" << pipeline->LastError() << "'");
  REQUIRE(pipeline->Running());

  const auto& stats = pipeline->Stats();
  INFO("p50=" << stats.P50() << "ms p99=" << stats.P99()
              << "ms drops=" << stats.Dropped());
  REQUIRE(stats.Count() >= 30);
  REQUIRE(stats.P99() < 250.0);
  // A barrier mistake surfaces as device removal, which the render loop reports
  // rather than throwing. Requiring the error to stay empty is what makes this
  // test worth running.
  REQUIRE(pipeline->LastError().empty());

  pipeline->Stop();
  REQUIRE(pipeline->Running() == false);
}

TEST_CASE("pipeline hides the overlay when the target window closes", "[device]") {
  auto gpu = DetectPrimaryGpu();
  REQUIRE(gpu.has_value());

  auto app = std::make_unique<PatternApp>();
  REQUIRE(app->Launch());

  PipelineConfig cfg;
  cfg.target = app->hwnd;
  auto pipeline = Pipeline::Create(*gpu, cfg, PassthroughPass::Create());
  REQUIRE(pipeline != nullptr);
  pipeline->Start();
  PumpFor(500ms);

  const HWND overlay = pipeline->OverlayHwnd();
  REQUIRE(IsWindowVisible(overlay));

  app.reset();  // target disappears

  bool hidden = false;
  for (int i = 0; i < 120 && !hidden; ++i) {
    hidden = !IsWindowVisible(overlay);
    if (!hidden) PumpFor(25ms);
  }
  // If this fails, the two useful questions are whether the render loop ever
  // noticed (lastError set) and whether it is still spinning (Running).
  INFO("lastError='" << pipeline->LastError() << "' running=" << pipeline->Running()
       << " frames=" << pipeline->Stats().Count());
  REQUIRE(hidden);   // spec failure rule: fail to a visible game, never to black

  pipeline->Stop();
}
