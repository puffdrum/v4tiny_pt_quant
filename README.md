# v4tiny_pt_quant
Quantization for yolo with xilinx/vitis-ai-pytorch

### Tool 
- Quantization tool: xilinx/vitis-ai:latest, [vitis-ai-pytorch](https://www.xilinx.com/html_docs/vitis_ai/1_3/pytorch.html#nvh1592318322520)
- Environment setup: [Vitis-AI](https://github.com/Xilinx/Vitis-AI)

### Model
- Model: yolov4-tiny pruned with [this project](https://github.com/tanluren/yolov3-channel-and-layer-pruning) or other yolo models
- Dataset: coco2017, with darknet format
- Input format: .cfg + .pt/.weights

### Errors
- Error unfixed: `aten_op 'view' parse failed(the layout of activation of reshape is ambiguous)` when run with `python quant.py --quant_mode calib`
