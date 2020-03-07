# About the benchmark
The benchmark is used to test the adlik serving performance of different models. Before using the benchmark to test the 
performance of the runtime, you need to build the client, the binary, and compile the model.

## Installing prerequisites

- python3
- pip3

## Build and install packages

1. Build clients and serving binary and make client pip packages (see [README.md](../../README.md)).

2. Install clients pip package:

   ```sh
   pip3 install {dir_of_pip_package}/adlik_serving_api-0.0.0-py2.py3-none-any.whl
   ```

3. Install model_compiler:

   ```sh
   cd {Adlik_root_dir}/model_compiler
   pip3 install .
   ```

##  Compile the test models

1. Prepare model code and serving_model.json (If you don't know how to write, you can refer to the existing serving_model.json).
   
   ```sh
   cd {Adlik_root_dir}/benchmark/test
   mkdir model_name
   cd model_name
   ```
   
   Then put your prepared model and serving_model.json in the directory model_name.
   
2. Run the model code, and save the model in {Adlik_root_dir}/benchmark/test/model_name/model.

   ```sh
   cd {Adlik_root_dir}/benchmark/test/model_name
   python3 model.py
   ```

3. Compile the model and save the serving model.
    
   ```sh
   cd {Adlik_root_dir}/benchmark/src
   python3 compile_model.py
   ```
   
   In the compile_model.py you can also specify the files that need to be compiled.

##  Test the serving performance

1. Deploy a serving service:

   ```sh
   cd {dir_of_adlik_serving_binary}
   ./adlik_serving --model_base_path={model_serving_dir} --grpc_port={grpc_port} --http_port={http_port}
   ```
   
   Usually the adlik serving binary is in the directory {Adlik_root_dir}/bazel-bin/adlik_serving, the grpc_port can
   be set to 8500 and the http_port can be set to 8501. And It should be noted that the type of the compiled model is 
   the same as the type of the serving service
   
2. Run a client and do inference:
   
   ```sh
   cd {Adlik_root_dir}/benchmark/test/client
   python3 xxx_client.py --batch-size=128 path_image
   ``` 

   The log of serving and client will be saved in time_log.log.
   
3. Analyze inference results

   ```sh
   cd {Adlik_root_dir}/benchmark/src
   python3 test_result.py path_client_log path_serving_log batch_size model_name runtime
   ```    
   
   Then you can get the performance analysis results of the serving.