from onnxsim import simplify
import onnx
import torch

model_path = 'D:\Sayyidan\CoolYeay\FastestDet_Tooth_Private\deploy\Yolov8\YOLOv8-onnx\weights\yolov8_640.pt'
#model = torch.load(model_path,map_location=torch.device('cpu'))
img='D:\Sayyidan\CoolYeay\FastestDet_Tooth_Private\deploy\Yolov8\YOLOv8-onnx\karang gigi.jpg'\

device = torch.device("cpu") 
model.load_state_dict(torch.load(model_path, map_location=device))
    #sets the module in eval node
model.eval()




torch.onnx.export(model,                     # model being run
                          img,                       # model input (or a tuple for multiple inputs)
                          "yolov8.onnx",       # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization
# onnx-sim
onnx_model = onnx.load("yolov8.onnx")  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
print("onnx sim sucess...")
onnx.save(model_simp, "yolov8-sim.onnx")   