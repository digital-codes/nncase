from test_runner import *
import os
import nncase
import numpy as np

class Inference:
    def run_inference(self, cfg, case_dir, import_options, compile_options, model_content, preprocess_opt):
        names, args = self.split_value(cfg.infer)
        for combine_args in product(*args):
            dict_args = dict(zip(names, combine_args))
            if dict_args['ptq'] and len(self.inputs) != 1:
                continue
            if cfg.compile_opt.dump_import_op_range and len(self.inputs) != 1:
                continue
            infer_output_paths = self.nncase_infer(
                cfg, case_dir, import_options,
                compile_options, model_content, dict_args, preprocess_opt)
            judge, result = self.compare_results(
                self.output_paths, infer_output_paths, dict_args)
            assert(judge), 'Fault result in infer' + result

    def nncase_infer(self, cfg, case_dir: str,
                     import_options: nncase.ImportOptions,
                     compile_options: nncase.CompileOptions,
                     model_content: Union[List[bytes], bytes],
                     kwargs: Dict[str, str],
                     preprocess: Dict[str, str]
                     ) -> List[Tuple[str, str]]:
        infer_dir = self.kwargs_to_path(
            os.path.join(case_dir, 'infer'), kwargs)
        compile_options = self.get_infer_compile_options(infer_dir, cfg, compile_options, kwargs, preprocess)
        compiler = nncase.Compiler(compile_options)
        self.import_model(compiler, model_content, import_options)
        self.set_infer_quant_opt(cfg, kwargs, preprocess, compiler)
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(os.path.join(infer_dir, 'test.kmodel'), 'wb') as f:
            f.write(kmodel)
            # todo:refactor
        sim = nncase.Simulator()
        sim.load_model(kmodel)
        self.set_infer_input(preprocess, case_dir, sim)
        sim.run()
        infer_output_paths = self.dump_infer_output(infer_dir, preprocess, sim)
        return infer_output_paths

    def get_infer_compile_options(self, infer_dir: str, cfg, compile_options: nncase.CompileOptions,
                     kwargs: Dict[str, str],
                     preprocess: Dict[str, str]):
        compile_options.target = kwargs['target']
        compile_options.dump_dir = infer_dir
        compile_options.dump_asm = cfg.compile_opt.dump_asm
        compile_options.dump_ir = cfg.compile_opt.dump_ir
        compile_options.dump_quant_error = cfg.compile_opt.dump_quant_error
        compile_options.dump_import_op_range = cfg.compile_opt.dump_import_op_range
        compile_options.is_fpga = cfg.compile_opt.is_fpga
        compile_options.use_mse_quant_w = cfg.compile_opt.use_mse_quant_w
        compile_options.input_type = preprocess['input_type']
        compile_options.quant_type = cfg.compile_opt.quant_type
        compile_options.w_quant_type = cfg.compile_opt.w_quant_type
        compile_options.swapRB = preprocess['swapRB']
        compile_options.input_shape = self.pre_process[3]['input_shape'] if self.pre_process[3]['input_shape'] != [
        ] else self.pre_process[3]['model_shape']
        compile_options.input_range = preprocess['input_range']
        compile_options.preprocess = preprocess['preprocess']
        compile_options.mean = preprocess['mean']
        compile_options.std = preprocess['std']
        compile_options.input_layout = preprocess['input_layout']
        compile_options.output_layout = preprocess['output_layout']
        compile_options.tcu_num = cfg.compile_opt.tcu_num
        return compile_options

    def set_infer_quant_opt(self, cfg, kwargs, preprocess, compiler):
        if cfg.compile_opt.dump_import_op_range:
            dump_range_options = nncase.DumpRangeTensorOptions()
            dump_range_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.dump_range_data]).tobytes())
            dump_range_options.samples_count = cfg.generate_dump_range_data.batch_size
            compiler.dump_range_options(dump_range_options)
        if kwargs['ptq']:
            ptq_options = nncase.PTQTensorOptions()
            ptq_options.set_tensor_data(np.asarray(
                [self.transform_input(sample['data'], preprocess['input_type'], "infer") for sample in self.calibs]).tobytes())
            ptq_options.samples_count = cfg.generate_calibs.batch_size
            compiler.use_ptq(ptq_options)

    def set_infer_input(self, preprocess, case_dir, sim):
        for i in range(len(self.inputs)):
            data = self.transform_input(self.inputs[i]['data'], preprocess['input_type'], "infer")
            dtype = preprocess['input_type']
            if preprocess['preprocess'] and dtype != 'float32':
                data.tofile(os.path.join(case_dir, f'input_{i}_{dtype}.bin'))
                self.totxtfile(os.path.join(case_dir, f'input_{i}_{dtype}.txt'), data)

            sim.add_input_tensor(nncase.RuntimeTensor.from_numpy(data))

    def dump_infer_output(self, infer_dir, preprocess, sim):
        infer_output_paths = []
        for i, output in enumerate(sim.all_numpy_output()):
            if preprocess['preprocess'] and len(output.shape) == 4:
                if(preprocess['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                    output = np.transpose(output, [0, 3, 1, 2])
                elif (preprocess['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                    output = np.transpose(output, [0, 2, 3, 1])
            infer_output_paths.append((
                os.path.join(infer_dir, f'nncase_result_{i}.bin'),
                os.path.join(infer_dir, f'nncase_result_{i}.txt')))
            output.tofile(infer_output_paths[-1][0])
            self.totxtfile(infer_output_paths[-1][1], output)
        return infer_output_paths