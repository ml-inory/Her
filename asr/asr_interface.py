import abc


class ASRInterface(metaclass=abc.ABCMeta):
    """
    Abstract class of ASR(Speech to Text)
    
    """
    
    @abc.abstractmethod
    def initialize(self, config: dict):
        """
        Initialize ASR engine.Do things like setting vad config, loading models, etc.

        config: yaml config 
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def set_config(self, config: dict):
        """
        Set config of ASR

        config: yaml config
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def samplerate(self) -> int:
        raise NotImplementedError
    

    @abc.abstractmethod
    def run(self, audio_data: dict) -> dict:
        """
        Run VAD

        audio_data: {
            start_timestamp[optional]: timestamp in milliseconds, yyyy/mm/dd HH:MM:SS.f,
            end_timestamp[optional]: timestamp in milliseconds, yyyy/mm/dd HH:MM:SS.f,
            data: np.ndarray[float]
        }

        return: {
            start_timestamp[optional]: timestamp in milliseconds, yyyy/mm/dd HH:MM:SS.f,
            end_timestamp[optional]: timestamp in milliseconds, yyyy/mm/dd HH:MM:SS.f,
            text: str
        }
        """
        raise NotImplementedError