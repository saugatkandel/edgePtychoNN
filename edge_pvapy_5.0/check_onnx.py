import onnx
import tensorrt as trt

from onnx import helper, shape_inference
from onnx import TensorProto

MODEL_PATHS = {
    "128": "/home/beams/ABABU/ptychoNN-test/new_models/training4_1.8khz/ptychoNN_8.onnx",
    "512": "/home/beams/SKANDEL/code/anakha_ptychoNN-test/models_02_10_23/ptychoNN_8.onnx",
}

CURRENT_MODEL = "512"


def basic_check(model, print_graph: bool = False):

    # Check that the model is well formed
    onnx.checker.check_model(model)

    if print_graph:
        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))


def shape_check(model):
    inferred_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)


def engine_build_from_onnx(onnx_mdl):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.max_workspace_size = 1 * (1 << 30)  # the maximum size that any layer in the network can use

    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    success = parser.parse_from_file(onnx_mdl)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        return None

    return builder.build_engine(network, config)


if __name__ == "__main__":
    # Load the ONNX model
    model = onnx.load(MODEL_PATHS[CURRENT_MODEL])
    print("Basic check...")
    basic_check(model)
    print("Passed...")
    print("Shape check...")
    shape_check(model)
