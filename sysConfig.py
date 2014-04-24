import logging
class sysConfig:
    def logConfig(self):
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s')
        return