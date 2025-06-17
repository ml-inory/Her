import abc


class VADInterface(metaclass=abc.ABCMeta):
    """
    Abstract class of VAD(Voice activity detection)
    
    """
    
    @abc.abstractmethod
    def initialize(self, config):
        """
        Initialize VAD engine.Do things like setting vad config, loading models, etc.

        config: yaml config 
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def set_config(self, config):
        """
        Set config of VAD

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
            samplerate: int,
            data: np.ndarray[float]
        }

        return: {
            samplerate: int,
            start_timestamp: timestamp in milliseconds, yyyy/mm/dd HH:MM:SS.f,
            end_timestamp: timestamp in milliseconds, yyyy/mm/dd HH:MM:SS.f,
            data: np.ndarray[float]
        }
        """
        raise NotImplementedError