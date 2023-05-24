import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np
import pvapy as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor

CURRENT_PATH = os.getcwd()

BATCH_SIZE = 8

OUTPUT_NX, OUTPUT_NY = 128, 128


MODEL_INPUT_SIZE = 512
QUANTIZATION = False

MODEL_PATHS = {
    "128": "/home/beams/ABABU/ptychoNN-test/new_models/training4_1.8khz/ptychoNN_8.onnx",
    "512": f"/home/beams/SKANDEL/beamtime_data/sector26_02_28_23/ptychonn_02_28_23/iteration_03_02_04_00/best_model_bsz_{BATCH_SIZE}.onnx",
    "512_QUANTIZED": f"/home/beams0/SKANDEL/code/TensorRT/tools/pytorch-quantization/examples/quantized_nn_batch_{BATCH_SIZE}.onnx",
}


# I am not sure if this modification is  necessary
NGPUS = 4


class InferPtychoNNImageProcessor(AdImageProcessor):
    def __init__(self, configDict={}):
        AdImageProcessor.__init__(self, configDict)
        self.tq_frame_q = mp.Queue(maxsize=-1)
        self.batch_q = mp.Queue(maxsize=-1)
        self.frm_id_q = mp.Queue(maxsize=-1)
        self.nFramesProcessed = 0
        self.nBatchesProcessed = 0
        self.inferTime = 0

        self.outputDirectory = configDict.get("outputDirectory", None)
        self.inputFileSaveDirectory = configDict.get("inputFileSaveDirectory", None)

        if self.outputDirectory is not None:
            self.logger.debug(f"Using output directory: {self.outputDirectory}")
            if not os.path.exists(self.outputDirectory):
                self.logger.debug(f"Creating output directory: {self.outputDirectory}")
                os.makedirs(self.outputDirectory)
            self.outputFileNameFormat = configDict.get("outputFileNameFormat", "{uniqueId:06}.{processorId}.bl2")
            self.logger.debug(f"Using output file name format: {self.outputFileNameFormat}")
        else:
            self.logger.debug("Not saving output inferences.")

        if self.inputFileSaveDirectory is not None:
            self.logger.debug(f"Saving input files in directory: {self.inputFileSaveDirectory}")
            if not os.path.exists(self.inputFileSaveDirectory):
                self.logger.debug(f"Creating output directory: {self.inputFileSaveDirectory}")
                os.makedirs(self.inputFileSaveDirectory)
            self.inputFileNameFormat = configDict.get("inputFileNameFormat", "{uniqueId:06}.{processorId}.bl2")
            self.logger.debug(f"Using output file name format: {self.inputFileNameFormat}")
        else:
            self.logger.debug("Not saving output inferences.")

        self.bsz = configDict.get("bsz", BATCH_SIZE)

        model_name = f"{MODEL_INPUT_SIZE}"
        if QUANTIZATION:
            model_name += "_QUANTIZED"
        self.onnx_mdl = configDict.get("onnx_mdl", MODEL_PATHS[model_name])
        print(self.onnx_mdl)
        self.isDone = False

    def inferWorker(self):
        self.logger.debug("Starting infer worker")
        self.gpu = (self.processorId - 1) % NGPUS
        self.logger.debug(f"Using gpu: {self.gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        import sys

        sys.path.insert(0, CURRENT_PATH)
        from inferPtychoNN import inferPtychoNNtrt
        from codecAD import CodecAD
        import blosc2

        # import blosc

        # blosc.set_nthreads(1)

        self.codecAD = CodecAD()
        self.inferEngine = inferPtychoNNtrt(
            self,
            mbsz=self.bsz,
            onnx_mdl=self.onnx_mdl,
            tq_diff=self.batch_q,
            frm_id_q=self.frm_id_q,
            quantization=QUANTIZATION,
        )
        self.logger.debug(f"Created infer engine using mbsz={self.bsz} and onnx_mdl={self.onnx_mdl}")
        bsz = self.bsz
        batch_list = []
        frm_id_list = []

        waitTime = 1
        while True:
            if self.isDone:
                break
            try:
                frameId, data, ny, nx, codec, compressed, uncompressed = self.tq_frame_q.get(
                    block=True, timeout=waitTime
                )
                # frm_id, in_frame, ny, nx = self.tq_frame_q.get(block=True, timeout=waitTime)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.isDone = True
                break
            except EOFError:
                self.isDone = True
                break
            except Exception as ex:
                self.logger.error(f"Unexpected error caught: {ex} {type(ex)}")
                break

            if not codec["name"]:
                image = np.reshape(data, (ny, nx))
            else:
                self.codecAD.decompress(data, codec, compressed, uncompressed)
                decompressed = self.codecAD.getData()
                # decompressed = blosc.decompress(data)
                image = np.frombuffer(decompressed, dtype=np.float32).reshape((ny, nx))

            batch_list.append(image)
            frm_id_list.append(frameId)

            outputFilePath = None
            inputFilePath = None
            if self.outputDirectory is not None:
                outputFilePath = os.path.join(self.outputDirectory, self.outputFileNameFormat)
            if self.inputFileSaveDirectory is not None:
                inputFilePath = os.path.join(self.inputFileSaveDirectory, self.inputFileNameFormat)

            while len(batch_list) >= bsz and not self.isDone:
                batch_chunk = np.array(batch_list[:bsz]).astype(np.float32)
                batch_frm_id = np.array((frm_id_list[:bsz]))
                self.batch_q.put(batch_chunk)
                self.frm_id_q.put(batch_frm_id)
                batch_list = batch_list[bsz:]
                frm_id_list = frm_id_list[bsz:]

                t0 = time.time()
                self.inferEngine.batch_infer(
                    nx, ny, OUTPUT_NX, OUTPUT_NY, outputFilePath, inputFilePath, self.processorId
                )

                t1 = time.time()
                self.nBatchesProcessed += 1
                self.nFramesProcessed += bsz
                self.inferTime += t1 - t0

        try:
            self.logger.debug(f"Stopping infer engine")
            self.inferEngine.stop()
        except Exception as ex:
            self.logger.warn(f"Error stopping infer engine: {ex}")
        self.logger.debug("Infer worker is done")

    def start(self):
        self.inferThread = threading.Thread(target=self.inferWorker)
        self.inferThread.start()

    def stop(self):
        self.logger.debug("Signaling infer worker to stop")
        self.isDone = True

    def configure(self, configDict):
        self.logger.debug(f"Configuration update: {configDict}")
        if "outputDirectory" in configDict:
            outputDirectory = configDict.get("outputDirectory")
            self.logger.debug(f"Reconfigured output directory: {outputDirectory}")
            if not os.path.exists(outputDirectory):
                self.logger.debug(f"Creating output directory: {outputDirectory}")
                os.makedirs(outputDirectory)
            self.outputDirectory = outputDirectory
        if "outputFileNameFormat" in configDict:
            self.outputFileNameFormat = configDict.get("outputFileNameFormat")

    def process(self, pvObject):
        if self.isDone:
            return

        codec = pvObject["codec"]
        frameId = pvObject["uniqueId"]
        fieldKey = pvObject.getSelectedUnionFieldName()
        dims = pvObject["dimension"]
        nx = dims[0]["size"]
        ny = dims[1]["size"]

        data = pvObject["value"][0][fieldKey]
        compressed = pvObject["compressedSize"]
        uncompressed = pvObject["uncompressedSize"]

        self.tq_frame_q.put((frameId, data, ny, nx, codec, compressed, uncompressed))
        return pvObject

    def resetStats(self):
        self.nFramesProcessed = 0
        self.nBatchesProcessed = 0
        self.inferTime = 0

    # Retrieve statistics for user processor
    def getStats(self):
        inferRate = 0
        frameProcessingRate = 0
        if self.nBatchesProcessed > 0:
            inferRate = self.nBatchesProcessed / self.inferTime
            frameProcessingRate = self.nFramesProcessed / self.inferTime
        nFramesQueued = self.tq_frame_q.qsize()
        return {
            "nFramesProcessed": self.nFramesProcessed,
            "nBatchesProcessed": self.nBatchesProcessed,
            "nFramesQueued": nFramesQueued,
            "inferTime": self.inferTime,
            "inferRate": inferRate,
            "frameProcessingRate": frameProcessingRate,
        }

    # Define PVA types for different stats variables
    def getStatsPvaTypes(self):
        return {
            "nFramesProcessed": pva.UINT,
            "nBatchesProcessed": pva.UINT,
            "nFramesQueued": pva.UINT,
            "inferTime": pva.DOUBLE,
            "inferRate": pva.DOUBLE,
            "frameProcessingRate": pva.DOUBLE,
        }

    def getOutputPvObjectType(self, pvObject):
        return pva.NtNdArray()
