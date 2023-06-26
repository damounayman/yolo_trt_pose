#pragma once

#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>

using namespace pose;

class YoloV8Pose {
public:
    /**
     * @brief Constructor for the YoloV8Pose class.
     * @param enginePath Path to the TensorRT engine file.
     */
    explicit YoloV8Pose(const std::string& enginePath);

    /**
     * @brief Destructor for the YoloV8Pose class.
     */
    ~YoloV8Pose();

    /**
     * @brief Initializes the pipeline and loads the engine.
     * @param warmup Flag indicating whether to perform engine warmup or not.
     */
    void makePipe(bool warmup = true);

    /**
     * @brief Copies image data from a cv::Mat object to the input tensor.
     * @param image Input image in cv::Mat format.
     */
    void copyFromMat(const cv::Mat& image);

    /**
     * @brief Copies image data from a cv::Mat object to the input tensor and resizes it.
     * @param image Input image in cv::Mat format.
     * @param size Target size for resizing the image.
     */
    void copyFromMat(const cv::Mat& image, cv::Size& size);

    /**
     * @brief Resizes the input image using letterboxing.
     * @param image Input image in cv::Mat format.
     * @param out Output image in cv::Mat format.
     * @param size Target size for resizing the image.
     */
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    /**
     * @brief Performs inference using the loaded engine.
     */
    void infer();

    /**
     * @brief Performs post-processing on the inference output to get the detected objects and poses.
     * @param objs Vector to store the detected objects and poses.
     * @param scoreThreshold Score threshold for object filtering.
     * @param IOThreshold IoU threshold for non-maximum suppression.
     * @param topk Maximum number of objects to keep.
     */
    void postprocess(std::vector<Object>& objs, float scoreThreshold = 0.25f,
                     float IOThreshold = 0.65f, int topk = 100);

    /**
     * @brief Draws the detected objects and poses on the input image.
     * @param image Input image in cv::Mat format.
     * @param res Output image in cv::Mat format.
     * @param objs Vector of detected objects and poses.
     * @param SKELETON Skeleton definition for pose drawing.
     * @param KPS_COLORS Colors for keypoints.
     * @param LIMB_COLORS Colors for limbs.
     */
    static void drawObjects(const cv::Mat& image, cv::Mat& res,
                            const std::vector<Object>& objs,
                            const std::vector<std::vector<unsigned int>>& SKELETON,
                            const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                            const std::vector<std::vector<unsigned int>>& LIMB_COLORS);

private:
    int numInputs_ = 0;
    int numOutputs_ = 0;
    std::vector<Binding> inputBindings_;
    std::vector<Binding> outputBindings_;
    std::vector<void*> hostPtrs_;
    std::vector<void*> devicePtrs_;
    PreParam pparam_;
    int numBindings_;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t cudaStream_ = nullptr;
    Logger gLogger_{nvinfer1::ILogger::Severity::kERROR};
};

