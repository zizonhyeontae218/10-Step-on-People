from onebit_llm.web import create_app


if __name__ == "__main__":
    create_app().launch(inbrowser=True, server_name="127.0.0.1")
