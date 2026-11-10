"""Portable, optional process resource measurements."""
import sys


def peak_rss_kib_linux():
    """Return Linux process peak RSS, or None when the platform lacks it."""
    if sys.platform != 'linux':
        return None
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
