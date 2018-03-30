# -*- coding: utf-8 -*-

from logging import getLogger, StreamHandler, DEBUG


def create_logger(name):
    """
    ロガーを生成して返す
    :param name: ロガー名
    :return: ロガー
    """
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
