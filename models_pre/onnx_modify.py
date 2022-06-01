import onnx
from onnx import helper


if __name__ == "__main__":
    model = onnx.load_model('supercombo.onnx')

    interm_layer = helper.ValueInfoProto()
    interm_layer.name = '1030'  # direct access to lead car output

    model.graph.output.append(interm_layer)
    onnx.save(model, 'supercombo_test.onnx')

