// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Sample program demonstrating basic ONNX Runtime C API usage.
// Loads a simple ONNX model (C = A + B), runs inference, and prints the result.
//
// Generate the model first:  python generate_model.py

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

// Helper macro to check OrtStatus and exit on error
#define CHECK_ORT_STATUS(status)                                       \
  do {                                                                 \
    if (status != NULL) {                                             \
      fprintf(stderr, "ORT Error: %s\n", OrtGetErrorMessage(status)); \
      OrtReleaseStatus(status);                                       \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

int main(int argc, char* argv[]) {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "Failed to get ORT API\n");
    return EXIT_FAILURE;
  }

  // -----------------------------------------------------------------------
  // 1. Initialize the ONNX Runtime environment
  // -----------------------------------------------------------------------
  OrtEnv* env = NULL;
  OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_sample", &env);
  CHECK_ORT_STATUS(status);

  printf("ONNX Runtime version: %s\n\n", OrtGetVersionString());

  // -----------------------------------------------------------------------
  // 2. Create session options (could add execution providers here)
  // -----------------------------------------------------------------------
  OrtSessionOptions* session_options = NULL;
  status = g_ort->CreateSessionOptions(&session_options);
  CHECK_ORT_STATUS(status);

  status = g_ort->SetIntraOpNumThreads(session_options, 1);
  CHECK_ORT_STATUS(status);

  status = g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);
  CHECK_ORT_STATUS(status);

  // -----------------------------------------------------------------------
  // 3. Load the ONNX model from a file
  //    Generate with:  python generate_model.py
  // -----------------------------------------------------------------------
  const char* model_path = (argc > 1) ? argv[1] : "add_model.onnx";
  printf("Loading model: %s\n", model_path);

  OrtSession* session = NULL;
  status = g_ort->CreateSession(env, model_path, session_options, &session);
  CHECK_ORT_STATUS(status);

  // -----------------------------------------------------------------------
  // 4. Query model metadata: input/output names and shapes
  // -----------------------------------------------------------------------
  OrtAllocator* allocator = NULL;
  status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
  CHECK_ORT_STATUS(status);

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  
  status = g_ort->SessionGetInputCount(session, &num_inputs);
  CHECK_ORT_STATUS(status);
  
  status = g_ort->SessionGetOutputCount(session, &num_outputs);
  CHECK_ORT_STATUS(status);

  printf("Model inputs:  %zu\n", num_inputs);
  printf("Model outputs: %zu\n", num_outputs);

  // Collect input/output names
  char** input_names = (char**)malloc(num_inputs * sizeof(char*));
  char** output_names = (char**)malloc(num_outputs * sizeof(char*));

  for (size_t i = 0; i < num_inputs; ++i) {
    char* name = NULL;
    status = g_ort->SessionGetInputName(session, i, allocator, &name);
    CHECK_ORT_STATUS(status);
    printf("  Input  %zu: %s\n", i, name);
    input_names[i] = name;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    char* name = NULL;
    status = g_ort->SessionGetOutputName(session, i, allocator, &name);
    CHECK_ORT_STATUS(status);
    printf("  Output %zu: %s\n", i, name);
    output_names[i] = name;
  }
  printf("\n");

  // -----------------------------------------------------------------------
  // 5. Prepare input tensors
  // -----------------------------------------------------------------------
  // Our model expects two float tensors of shape [1, 3].
  int64_t batch_size = 1;
  int64_t num_elements = 3;
  int64_t input_shape[] = {batch_size, num_elements};
  size_t input_shape_len = 2;

  float input_a[] = {1.0f, 2.0f, 3.0f};
  float input_b[] = {4.0f, 5.0f, 6.0f};

  OrtMemoryInfo* memory_info = NULL;
  status = g_ort->CreateMemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault, &memory_info);
  CHECK_ORT_STATUS(status);

  OrtValue* tensor_a = NULL;
  status = g_ort->CreateTensorWithDataAsOrtValue(
      memory_info, (void*)input_a, sizeof(input_a),
      input_shape, input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor_a);
  CHECK_ORT_STATUS(status);

  OrtValue* tensor_b = NULL;
  status = g_ort->CreateTensorWithDataAsOrtValue(
      memory_info, (void*)input_b, sizeof(input_b),
      input_shape, input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor_b);
  CHECK_ORT_STATUS(status);

  OrtValue* input_tensors[] = {tensor_a, tensor_b};
  size_t input_tensors_len = 2;

  // -----------------------------------------------------------------------
  // 6. Run inference
  // -----------------------------------------------------------------------
  printf("Running inference...\n");

  OrtRunOptions* run_options = NULL;
  status = g_ort->CreateRunOptions(&run_options);
  CHECK_ORT_STATUS(status);

  OrtValue* output_tensors = NULL;
  status = g_ort->Run(session, run_options,
                      (const char* const*)input_names, input_tensors, input_tensors_len,
                      (const char* const*)output_names, num_outputs, &output_tensors);
  CHECK_ORT_STATUS(status);

  // -----------------------------------------------------------------------
  // 7. Process output
  // -----------------------------------------------------------------------
  if (!output_tensors) {
    fprintf(stderr, "No output tensors\n");
    exit(EXIT_FAILURE);
  }

  float* output_data = NULL;
  status = g_ort->GetTensorMutableData(output_tensors, (void**)&output_data);
  CHECK_ORT_STATUS(status);

  OrtTensorTypeAndShapeInfo* type_info = NULL;
  status = g_ort->GetTensorTypeAndShape(output_tensors, &type_info);
  CHECK_ORT_STATUS(status);

  size_t output_count = 0;
  status = g_ort->GetTensorShapeElementCount(type_info, &output_count);
  CHECK_ORT_STATUS(status);

  printf("\nInputs:\n");
  printf("  A = [");
  for (size_t i = 0; i < (size_t)num_elements; ++i) {
    printf("%s%.1f", (i ? ", " : ""), input_a[i]);
  }
  printf("]\n");

  printf("  B = [");
  for (size_t i = 0; i < (size_t)num_elements; ++i) {
    printf("%s%.1f", (i ? ", " : ""), input_b[i]);
  }
  printf("]\n");

  printf("\nOutput (A + B):\n");
  printf("  C = [");
  for (size_t i = 0; i < output_count; ++i) {
    printf("%s%.1f", (i ? ", " : ""), output_data[i]);
  }
  printf("]\n");

  // Verify correctness
  int correct = 1;
  for (size_t i = 0; i < (size_t)num_elements; ++i) {
    if (output_data[i] != input_a[i] + input_b[i]) {
      correct = 0;
      break;
    }
  }
  printf("\nResult: %s\n", correct ? "PASS" : "FAIL");

  // -----------------------------------------------------------------------
  // 8. Cleanup
  // -----------------------------------------------------------------------
  g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
  g_ort->ReleaseValue(output_tensors);
  g_ort->ReleaseRunOptions(run_options);
  g_ort->ReleaseMemoryInfo(memory_info);
  g_ort->ReleaseValue(tensor_a);
  g_ort->ReleaseValue(tensor_b);

  for (size_t i = 0; i < num_inputs; ++i) {
    g_ort->AllocatorFree(allocator, input_names[i]);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    g_ort->AllocatorFree(allocator, output_names[i]);
  }

  free(input_names);
  free(output_names);

  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);

  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
