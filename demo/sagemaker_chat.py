import gradio as gr
import boto3
import json
import io

# hyperparameters for llm
parameters = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.03,
    "stop": ["\nUser:", "<|endoftext|>", " User:", "###"],
}

system_prompt = "You are an helpful Assistant, called Falcon. Knowing everyting about AWS."


# Helper for reading lines from a stream
class LineIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


# helper method to format prompt
def format_prompt(message, history, system_prompt):
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n"
    for user_prompt, bot_response in history:
        prompt += f"User: {user_prompt}\n"
        prompt += f"Falcon: {bot_response}\n"  # Response already contains "Falcon: "
    prompt += f"""User: {message}
Falcon:"""
    return prompt


def create_gradio_app(
    endpoint_name,
    session=boto3,
    parameters=parameters,
    system_prompt=system_prompt,
    format_prompt=format_prompt,
    concurrency_count=4,
    share=True,
):
    smr = session.client("sagemaker-runtime")

    def generate(
        prompt,
        history,
    ):
        formatted_prompt = format_prompt(prompt, history, system_prompt)

        request = {"inputs": formatted_prompt, "parameters": parameters, "stream": True}
        resp = smr.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            Body=json.dumps(request),
            ContentType="application/json",
        )

        output = ""
        for c in LineIterator(resp["Body"]):
            c = c.decode("utf-8")
            if c.startswith("data:"):
                chunk = json.loads(c.lstrip("data:").rstrip("/n"))
                if chunk["token"]["special"]:
                    continue
                if chunk["token"]["text"] in request["parameters"]["stop"]:
                    break
                output += chunk["token"]["text"]
                for stop_str in request["parameters"]["stop"]:
                    if output.endswith(stop_str):
                        output = output[: -len(stop_str)]
                        output = output.rstrip()
                        yield output

                yield output
        return output

    demo = gr.ChatInterface(generate, title="Chat with Amazon SageMaker", chatbot=gr.Chatbot(layout="panel"))

    demo.queue(concurrency_count=concurrency_count).launch(share=share)
