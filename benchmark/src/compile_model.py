import os
import json
import model_compiler


def compile_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    test_model_dir = os.path.join(base_dir, "test", "test_model")
    for file in os.listdir(test_model_dir):
        request_dir = os.path.join(test_model_dir, file, "serving_model.json")
        try:
            with open(request_dir, 'r') as request_file:
                request = json.load(request_file)
                model_dir = request["input_model"]
                request["input_model"] = os.path.join(test_model_dir, file, model_dir)
                export_dir = request["export_path"]
                request["export_path"] = os.path.join(test_model_dir, file, export_dir)
                result = model_compiler.compile_model(request)
                print(result)
        except FileNotFoundError:
            print(f"Can not compile the model in {os.path.join(test_model_dir, file)}")


if __name__ == '__main__':
    compile_model()
