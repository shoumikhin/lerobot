// (c) 2025 HuggingFace Inc. and affiliates. Apache-2.0.
//
// Reference C++ runner for lerobot AOTI export bundles.
// Loads a .pt2 package via AOTIModelPackageLoader and executes the
// prefill + Euler decode loop described in manifest.json.
//
// This is a TEMPLATE. Adapt it to your robot's sensor/actuator interface.
// Compile against the libtorch that matches the AOTI export's torch version.
//
// Build:
//   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch -DBUNDLE_DIR=/path/to/bundle
//   cmake --build build
//
// Run:
//   ./build/lerobot_runner /path/to/bundle <instruction_text>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal JSON reader for manifest.json (replace with nlohmann/json etc.)
// ---------------------------------------------------------------------------

static int read_n_steps(const std::string& bundle_dir) {
    // Phase 1: parse manifest.json["decode"]["n_steps"].
    // For now, return the default SmolVLA value.
    return 10;
}

// ---------------------------------------------------------------------------
// Preprocessing stubs (Phase 1: implement from manifest.preprocess)
// ---------------------------------------------------------------------------

static std::vector<torch::Tensor> preprocess_images(
    const std::string& bundle_dir,
    /* your camera data here */) {
    // TODO Phase 1: resize+pad images to (H, W) from manifest.preprocess.image.
    // Normalize to [-1, 1], convert to NCHW float32.
    throw std::runtime_error("preprocess_images: Phase 1 TODO");
}

static torch::Tensor tokenize_instruction(
    const std::string& bundle_dir,
    const std::string& instruction) {
    // TODO Phase 1: run the tokenizer specified in manifest.preprocess.tokenizer.
    // Use HuggingFace tokenizers C++ or SentencePiece.
    throw std::runtime_error("tokenize_instruction: Phase 1 TODO");
}

static torch::Tensor normalize_state(
    const std::string& bundle_dir,
    const torch::Tensor& raw_state) {
    // TODO Phase 1: load mean/std from normalization.safetensors, apply (x - mean) / std.
    throw std::runtime_error("normalize_state: Phase 1 TODO");
}

static torch::Tensor unnormalize_action(
    const std::string& bundle_dir,
    const torch::Tensor& action_normalized) {
    // TODO Phase 1: load mean/std from normalization.safetensors, apply x * std + mean.
    throw std::runtime_error("unnormalize_action: Phase 1 TODO");
}

// ---------------------------------------------------------------------------
// Main runner
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <bundle_dir> <instruction>\n";
        return 1;
    }
    const std::string bundle_dir = argv[1];
    const std::string instruction = argv[2];

    // 1. Load the two AOTI packages.
    auto prefill_loader = torch::inductor::AOTIModelPackageLoader(
        bundle_dir + "/prefill.pt2");
    auto denoise_loader = torch::inductor::AOTIModelPackageLoader(
        bundle_dir + "/denoise_step.pt2");

    // 2. Preprocess (Phase 1 TODO: replace stubs with real implementations).
    auto images = preprocess_images(bundle_dir);       // list[Tensor]
    auto lang_tokens = tokenize_instruction(bundle_dir, instruction);
    auto lang_masks = torch::ones_like(lang_tokens).to(torch::kBool);
    auto raw_state = torch::zeros({1, 32});             // TODO: read from robot
    auto state = normalize_state(bundle_dir, raw_state);
    auto img_masks = std::vector<torch::Tensor>{torch::ones({1}, torch::kBool)};

    // 3. GRAPH A: prefill -> KV cache + prefix_pad_masks.
    std::vector<torch::Tensor> prefill_inputs;
    for (auto& img : images) prefill_inputs.push_back(img);
    for (auto& m : img_masks) prefill_inputs.push_back(m);
    prefill_inputs.push_back(lang_tokens);
    prefill_inputs.push_back(lang_masks);
    prefill_inputs.push_back(state);
    auto prefill_outputs = prefill_loader.run(prefill_inputs);
    auto prefix_pad_masks = prefill_outputs[0];
    // KV cache tensors: prefill_outputs[1..end]

    // 4. Decode loop: sample noise, run GRAPH B n_steps times.
    int n_steps = read_n_steps(bundle_dir);
    float dt = -1.0f / n_steps;
    auto x_t = torch::randn({1, 50, 32});  // noise (Phase 1: read shape from manifest)

    for (int step = 0; step < n_steps; ++step) {
        float t = 1.0f + step * dt;
        auto timestep = torch::tensor({t});

        std::vector<torch::Tensor> denoise_inputs = {x_t, timestep, prefix_pad_masks};
        for (size_t i = 1; i < prefill_outputs.size(); ++i)
            denoise_inputs.push_back(prefill_outputs[i]);

        auto v_t = denoise_loader.run(denoise_inputs)[0];
        x_t = x_t + dt * v_t;
    }

    // 5. Postprocess and emit the action.
    auto action = unnormalize_action(bundle_dir, x_t);
    std::cout << "Action (first step): " << action[0][0] << "\n";

    return 0;
}
