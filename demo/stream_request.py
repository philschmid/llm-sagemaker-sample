import gradio as gr
import boto3
import json
import io


parameters = {
    "stream": True,
    "top_p": 0.9,
    "temperature": 1.0,
    "max_tokens": 16_000,
}


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


def create_streamer(
    endpoint_name,
    session=boto3,
    parameters=parameters,
    system_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
):
    smr = session.client("sagemaker-runtime")

    def generate(prompt):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = smr.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            Body=json.dumps(
                {
                    "messages": messages,
                    **parameters,
                }
            ),
            ContentType="application/json",
        )

        output = ""
        for c in LineIterator(resp["Body"]):
            c = c.decode("utf-8")
            if c.startswith("data:"):
                chunk = json.loads(c.lstrip("data:").rstrip("/n"))["choices"][0]
                if chunk["finish_reason"]:
                    break
                output += chunk["delta"]["content"]
                yield output
        return output

    return generate
