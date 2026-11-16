# Bypass Windows Python 3.12 WMI hanging bug in platform._wmi_query
import sys
if sys.platform == "win32":
    try:
        import platform
        def _no_wmi(*args, **kwargs):
            raise OSError("WMI query disabled to prevent Windows hang")
        platform._wmi_query = _no_wmi
    except Exception:
        pass
