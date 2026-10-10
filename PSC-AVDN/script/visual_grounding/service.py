"""Compatibility CLI: python -m visual_grounding.service."""
if __name__ == "__main__":
    from andh_core.tools.grounding.service import main
    main()
else:
    import sys
    from andh_core.tools.grounding import service as _implementation
    sys.modules[__name__] = _implementation
