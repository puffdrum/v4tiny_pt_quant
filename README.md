# v4tiny_pt_quant
quantization for yolo with xilinx/vitis-ai-pytorch

Quantization tool: xilinx/vitis-ai:latest, vitis-ai-pytorch
Dataset: coco2017
Model: yolov4-tiny pruned with [this project](https://github.com/tanluren/yolov3-channel-and-layer-pruning)

Error unfixed: `aten_op 'view' parse failed(the layout of activation of reshape is ambiguous)`
