import logging


class Logger:
    """
    A class to add logging to a tree.
    """

    def __init__(
        self,
        name: str,
        logging_enabled: bool,
        section_name_decorator: str = "=",
        max_section_name_length: int = 100,
    ):
        self.logging_enabled = logging_enabled
        self.logger = self._init_logger(name=name)

        self.section_name_decorator = section_name_decorator
        self.max_section_name_length = max_section_name_length

    def _init_logger(self, name: str) -> logging.Logger:
        """
        Initialize a logger.
        :param name: The name of the logger.
        :return: The initialized logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        return logger

    def log_section(self, section_name: str, add_newline: bool = True):
        section_name_length = len(section_name)

        if section_name_length >= self.max_section_name_length:
            return self.log(section_name + "\n")

        remaining_length = self.max_section_name_length - section_name_length
        decorator_count = int(remaining_length // 2) - 1

        decorator_string = self.section_name_decorator * decorator_count

        self.log(f"{decorator_string} {section_name} {decorator_string}" + ("\n" if add_newline else ""))

    def log(self, message: str):
        """
        Log a message.
        :param message: The message to log.
        """
        if self.logging_enabled:
            self.logger.info(message)
