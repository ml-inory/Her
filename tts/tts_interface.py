import abc


class TTSInterface(metaclass=abc.ABCMeta):
    """
    Abstract class of TTS(Text to Speech)
    
    """
    
    @abc.abstractmethod
    def initialize(self, config: dict):
        """
        Initialize TTS engine.Do things like setting vad config, loading models, etc.

        config: yaml config 
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def set_config(self, config: dict):
        """
        Set config of TTS

        config: yaml config
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def run(self, text: str) -> dict:
        """
        Run TTS

        return: {
            audio: audio[np.ndarray],
            samplerate: int[16000]
        }
        """
        raise NotImplementedError