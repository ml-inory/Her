from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


def create_engines(config_file: str):
    import yaml
    from vad import VADFactory
    from asr import ASRFactory
    from llm import LLMFactory
    from tts import TTSFactory
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    vad_engine = VADFactory.create(config)
    asr_engine = ASRFactory.create(config)
    llm_engine = LLMFactory.create(config)
    tts_engine = TTSFactory.create(config)
    return [vad_engine, asr_engine, llm_engine, tts_engine]


class ASRParam(BaseModel):
    samplerate: int
    data: list[float]


class LLMParam(BaseModel):
    message: str


class TTSParam(BaseModel):
    text: str


vad_engine, asr_engine, llm_engine, tts_engine = create_engines("config.yaml")
app = FastAPI()


@app.post("/asr/")
async def run_asr(asr_param: ASRParam):
    global vad_engine, asr_engine

    vad_result = vad_engine.run({
        "samplerate": asr_param.samplerate,
        "data": [np.array(asr_param.data, dtype=np.float32)]
    })
    if vad_result is None:
        return {"status": "listening", "code": 204}
    
    asr_result = asr_engine.run(vad_result)
    response = asr_result
    response["status"] = "finish"
    response["code"] = 200
    return response


@app.post("/llm/")
async def run_llm(param: LLMParam):
    global llm_engine

    llm_response = llm_engine.run(param.message)
    return {
        "status": "finish",
        "code": 200,
        "text": llm_response
    }


@app.post("/tts/")
async def run_tts(param: TTSParam):
    global tts_engine

    audio_data = tts_engine.run(param.text)
    return {
        "status": "finish",
        "code": 200,
        "audio": audio_data["audio"].tolist(),
        "samplerate": audio_data["samplerate"]
    }