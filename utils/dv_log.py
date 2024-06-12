import json
import logging


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger_name': record.name,
            'module': record.module,
            'function': record.funcName,
            'message': record.getMessage()
        }
        return json.dumps(log_data)


class DVLogger():
    def __init__(self, logger_name, log_filename):
        self.file_handler = logging.FileHandler(log_filename)
        self.file_handler.setLevel(logging.INFO)
        self.json_formatter = JSONFormatter()
        self.file_handler.setFormatter(self.json_formatter)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.file_handler)

    # To log string
    def log(self, message):
        self.logger.info(message)

    # To log dictionary (agent reponse)
    def log_json(self, message):
        self.logger.info(json.dumps(message))
    
    def close(self):
        self.file_handler.close()
        self.logger.removeHandler(self.file_handler)