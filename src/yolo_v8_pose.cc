#include "yolo_v8_pose.h"

YoloV8Pose::YoloV8Pose(const std::string &enginePath) {
  // Step 1: Load serialized CUDA engine from file
  std::ifstream file(enginePath, std::ios::binary);
  assert(file.good());

  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engineData(size);
  assert(engineData.data());

  file.read(engineData.data(), size);
  file.close();

  // Step 2: Initialize TensorRT and create CUDA engine
  initLibNvInferPlugins(&gLogger_, "");
  runtime_ = nvinfer1::createInferRuntime(gLogger_);
  assert(runtime_ != nullptr);

  engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
  assert(engine_ != nullptr);

  // Step 3: Create the execution context and CUDA stream
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr);

  cudaStreamCreate(&cudaStream_);

  // Step 4: Populate the bindings information
  numBindings_ = engine_->getNbBindings();

  for (int i = 0; i < numBindings_; ++i) {
    Binding binding;
    nvinfer1::Dims dimensions;
    const nvinfer1::DataType dtype = engine_->getBindingDataType(i);
    const std::string name = engine_->getBindingName(i);
    binding.name = name;
    binding.dataSize = type_to_size(dtype);

    const bool isInput = engine_->bindingIsInput(i);
    dimensions = (isInput) ? engine_->getProfileDimensions(
                                 i, 0, nvinfer1::OptProfileSelector::kMAX)
                           : context_->getBindingDimensions(i);

    binding.size = get_size_by_dims(dimensions);
    binding.dimensions = dimensions;

    if (isInput) {
      numInputs_ += 1;
      inputBindings_.push_back(binding);
      // Set the maximum optimization shape for input bindings
      context_->setBindingDimensions(i, dimensions);
    } else {
      outputBindings_.push_back(binding);
      numOutputs_ += 1;
    }
  }
}

YoloV8Pose::~YoloV8Pose() {
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
  cudaStreamDestroy(cudaStream_);
  for (auto &ptr : devicePtrs_) {
    CHECK(cudaFree(ptr));
  }

  for (auto &ptr : hostPtrs_) {
    CHECK(cudaFreeHost(ptr));
  }
}

void YoloV8Pose::makePipe(bool warmup) {
  // Allocate device memory for input bindings
  for (auto &inputBinding : inputBindings_) {
    void *devicePtr;
    CHECK(cudaMalloc(&devicePtr, inputBinding.size * inputBinding.dataSize));
    devicePtrs_.push_back(devicePtr);
  }

  // Allocate device and host memory for output bindings
  for (auto &outputBinding : outputBindings_) {
    void *devicePtr;
    void *hostPtr;
    const size_t size = outputBinding.size * outputBinding.dataSize;
    CHECK(cudaMalloc(&devicePtr, size));
    CHECK(cudaHostAlloc(&hostPtr, size, 0));
    devicePtrs_.push_back(devicePtr);
    hostPtrs_.push_back(hostPtr);
  }

  if (warmup) {
    // Warm up the model by performing inference with dummy data
    for (int i = 0; i < 10; ++i) {
      for (const auto &inputBinding : inputBindings_) {
        const size_t size = inputBinding.size * inputBinding.dataSize;
        void *hostPtr = malloc(size);
        memset(hostPtr, 0, size);
        CHECK(cudaMemcpyAsync(devicePtrs_[0], hostPtr, size,
                              cudaMemcpyHostToDevice, cudaStream_));
        free(hostPtr);
      }
      infer();
    }
    printf("Model warmed up 10 times.\n");
  }
}

void YoloV8Pose::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {
  const float inp_h = size.height;
  const float inp_w = size.width;
  float height = image.rows;
  float width = image.cols;

  float r = std::min(inp_h / height, inp_w / width);
  int padw = std::round(width * r);
  int padh = std::round(height * r);

  cv::Mat tmp;
  if ((int)width != padw || (int)height != padh) {
    cv::resize(image, tmp, cv::Size(padw, padh));
  } else {
    tmp = image.clone();
  }

  float dw = inp_w - padw;
  float dh = inp_h - padh;

  dw /= 2.0f;
  dh /= 2.0f;
  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));

  cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                     {114, 114, 114});

  cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0),
                         true, false, CV_32F);
  pparam_.ratio = 1 / r;
  pparam_.dw = dw;
  pparam_.dh = dh;
  pparam_.height = height;
  pparam_.width = width;
  ;
}

void YoloV8Pose::copyFromMat(const cv::Mat &image) {
  cv::Mat nchw;
  auto &in_binding = inputBindings_[0];
  auto width = in_binding.dimensions.d[3];
  auto height = in_binding.dimensions.d[2];
  cv::Size size{width, height};
  letterbox(image, nchw, size);

  context_->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

  CHECK(cudaMemcpyAsync(devicePtrs_[0], nchw.ptr<float>(),
                        nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                        cudaStream_));
}

void YoloV8Pose::copyFromMat(const cv::Mat &image, cv::Size &size) {
  cv::Mat nchw;
  letterbox(image, nchw, size);
  context_->setBindingDimensions(
      0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
  CHECK(cudaMemcpyAsync(devicePtrs_[0], nchw.ptr<float>(),
                        nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                        cudaStream_));
}

void YoloV8Pose::infer() {
  // Enqueue the inference task on the execution context with the device
  // pointers
  context_->enqueueV2(devicePtrs_.data(), cudaStream_, nullptr);
  // Copy the output data from device to host
  for (int i = 0; i < numOutputs_; ++i) {
    const size_t outputSize =
        outputBindings_[i].size * outputBindings_[i].dataSize;
    CHECK(cudaMemcpyAsync(hostPtrs_[i], devicePtrs_[i + numInputs_], outputSize,
                          cudaMemcpyDeviceToHost, cudaStream_));
  }
  // Synchronize the CUDA stream to ensure completion of all operations
  cudaStreamSynchronize(cudaStream_);
}

void YoloV8Pose::postprocess(std::vector<Object> &objs, float scoreThreshold,
                             float IOThreshold, int topk) {
  objs.clear();
  // Get the dimensions of the output tensor
  const int numChannels = outputBindings_[0].dimensions.d[1];
  const int numAnchors = outputBindings_[0].dimensions.d[2];

  // Retrieve parameters for bounding box calculation
  const float &dw = pparam_.dw;
  const float &dh = pparam_.dh;
  const float &width = pparam_.width;
  const float &height = pparam_.height;
  const float &ratio = pparam_.ratio;

  // Initialize vectors to store detected objects and related information
  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;
  std::vector<std::vector<float>> kpss;
  // Get a pointer to the output tensor and transpose it for efficient access
  cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F,
                           static_cast<float *>(hostPtrs_[0]))
                       .t();
  // Reserve memory for vectors to avoid reallocations
  bboxes.reserve(numAnchors);
  scores.reserve(numAnchors);
  labels.resize(numAnchors, 0);  // Assuming a single class
  kpss.reserve(numAnchors);

 for (int i = 0; i < numAnchors; ++i) {
  const float* rowPtr = output.ptr<float>(i);
  const float* bboxesPtr = rowPtr;
  const float* scoresPtr = rowPtr + 4;
  const float* kpsPtr = rowPtr + 5;

  const float score = *scoresPtr;

  if (score > scoreThreshold) {
    const float x = *bboxesPtr++ - dw;
    const float y = *bboxesPtr++ - dh;
    const float w = *bboxesPtr++;
    const float h = *bboxesPtr;

    const float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
    const float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
    const float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
    const float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

    const cv::Rect bbox(x0, y0, x1 - x0, y1 - y0);

    std::vector<float> kps(17 * 3);
    const float* kpsPtrOrig = kpsPtr;  // Store the original pointer

    for (int k = 0; k < 17; ++k) {
      const float kps_x = (*kpsPtr++ - dw) * ratio;
      const float kps_y = (*kpsPtr++ - dh) * ratio;
      const float kps_s = *kpsPtr++;
      const int kpsIndex = 3 * k;
      kps[kpsIndex] = clamp(kps_x, 0.f, width);
      kps[kpsIndex + 1] = clamp(kps_y, 0.f, height);
      kps[kpsIndex + 2] = kps_s;
    }

    bboxes.push_back(bbox);
    scores.push_back(score);
    kpss.push_back(std::move(kps));

    kpsPtr = kpsPtrOrig;  // Reset the pointer to the original position for the next iteration
  }
}

#ifdef BATCHED_NMS
  cv::dnn::NMSBoxesBatched(bboxes, scores, labels, scoreThreshold, IOThreshold,
                           indices);
#else
  cv::dnn::NMSBoxes(bboxes, scores, scoreThreshold, IOThreshold, indices);
#endif

  // Add the top-k objects to the output vector
  const int numObjects = std::min(static_cast<int>(indices.size()), topk);
  objs.reserve(numObjects);
  for (int i = 0; i < numObjects; ++i) {
    const int index = indices[i];
    Object obj;
    obj.rect = bboxes[index];
    obj.prob = scores[index];
    obj.label = labels[index];
    obj.kps = std::move(kpss[index]);
    objs.push_back(std::move(obj));
  }
}

void YoloV8Pose::drawObjects(
    const cv::Mat &image, cv::Mat &res, const std::vector<Object> &objs,
    const std::vector<std::vector<unsigned int>> &SKELETON,
    const std::vector<std::vector<unsigned int>> &KPS_COLORS,
    const std::vector<std::vector<unsigned int>> &LIMB_COLORS) {
  res = image.clone();
  const int num_point = 17;
  for (auto &obj : objs) {
    cv::rectangle(res, obj.rect, {0, 0, 255}, 2);

    char text[256];
    sprintf(text, "person %.1f%%", obj.prob * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    int x = (int)obj.rect.x;
    int y = (int)obj.rect.y + 1;

    if (y > res.rows)
      y = res.rows;

    cv::rectangle(
        res, cv::Rect(x, y, label_size.width, label_size.height + baseLine),
        {0, 0, 255}, -1);

    cv::putText(res, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

    auto &kps = obj.kps;
    for (int k = 0; k < num_point + 2; k++) {
      if (k < num_point) {
        int kps_x = std::round(kps[k * 3]);
        int kps_y = std::round(kps[k * 3 + 1]);
        float kps_s = kps[k * 3 + 2];
        if (kps_s > 0.5f) {
          cv::Scalar kps_color =
              cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
          cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
        }
      }
      auto &ske = SKELETON[k];
      int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
      int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

      int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
      int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

      float pos1_s = kps[(ske[0] - 1) * 3 + 2];
      float pos2_s = kps[(ske[1] - 1) * 3 + 2];

      if (pos1_s > 0.5f && pos2_s > 0.5f) {
        cv::Scalar limb_color =
            cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
        cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
      }
    }
  }
}
