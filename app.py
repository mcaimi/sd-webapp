#!/usr/bin/env python

# load libraries
import sys
try:
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    import starlette.status as status
except Exception as e:
    print(f"Caught exception: {e}")
    sys.exit(-1)

# Build UI
from libs.sd_ui_local import StableDiffusionUI
from libs.vars import GRADIO_CUSTOM_PATH

# build gradio ui object
sd_ui = StableDiffusionUI()
sd_ui.buildUi()

# build the application object
sd_app = FastAPI()

# add a root path
@sd_app.get("/")
async def get_root():
    # Redirect to the main Gradio App
    return RedirectResponse(url=GRADIO_CUSTOM_PATH, status_code=status.HTTP_302_FOUND)

# attach gradio app
sd_ui.registerFastApiEndpoint(sd_app)
