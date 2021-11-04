
class WebappException(Exception):
    pass

class BadRequest(WebappException):
    pass

class Locked(WebappException):
    pass

class InternalServerError(WebappException):
    pass

class Unauthorized(WebappException):
    pass

