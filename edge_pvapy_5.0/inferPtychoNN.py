import numpy as np
import threading
from PIL import Image
from helper import inference
import blosc2


class inferPtychoNNtrt:
    def __init__(self, pvapyProcessor, mbsz, onnx_mdl, tq_diff, frm_id_q, quantization=False):
        self.tq_diff = tq_diff
        self.mbsz = mbsz
        self.onnx_mdl = onnx_mdl
        self.pvapyProcessor = pvapyProcessor
        self.frm_id_q = frm_id_q
        self.quantization = quantization

        import tensorrt as trt
        from helper import engine_build_from_onnx, mem_allocation
        import pycuda.autoinit  # must be in the same thread as the actual cuda execution

        self.context = pycuda.autoinit.context
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl, quantization)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()

    def stop(self):
        try:
            self.context.pop()
        except Exception as ex:
            pass

    def batch_infer(self, nx, ny, output_nx=None, output_ny=None, outputFilePath="", inputFilePath="", processorId=""):

        output_nx = nx if output_nx is None else output_nx
        output_ny = ny if output_ny is None else output_ny
        in_mb = self.tq_diff.get()
        bsz, ny, nx = in_mb.shape
        frm_id_list = self.frm_id_q.get()
        np.copyto(self.trt_hin, in_mb.astype(np.float32).ravel())
        pred = np.array(
            inference(self.trt_context, self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, self.trt_stream)
        )

        pred = pred.reshape(bsz, output_nx * output_ny)
        for j in range(0, len(frm_id_list)):
            image = pred[j].reshape(output_ny, output_nx)
            frameId = int(frm_id_list[j])
            outputNtNdArray = self.pvapyProcessor.generateNtNdArray2D(frameId, image)

            if outputFilePath:
                outputFileName = outputFilePath.format(
                    frameId=frameId, uniqueId=frameId, objectId=frameId, processorId=processorId
                )
                # blosc2.save_array(image, outputFileName)
                im = Image.fromarray(image)
                im.save(outputFileName)
            if inputFilePath:
                inputFileName = inputFilePath.format(
                    frameId=frameId, uniqueId=frameId, objectId=frameId, processorId=processorId
                )
                im2 = Image.fromarray(in_mb[j])
                im2.save(inputFileName)
                # blosc2.save_array(in_mb[j], inputFileName)

            self.pvapyProcessor.updateOutputChannel(outputNtNdArray)
