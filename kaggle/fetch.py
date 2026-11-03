"""Download all outputs of a Kaggle job into runs/<name>/.

`kaggle kernels output` (CLI 1.7.4.5) only fetches the first page (~500 files); this pages through all of them.
Already-downloaded files are skipped, so it can be re-run.

Usage: python kaggle/fetch.py j2_steps            # kernel <user>/vlalab-j2_steps -> runs/j2_steps/
Needs the Python that has the kaggle package (the one behind the `kaggle` CLI).
"""
import os
import sys

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push import KERNEL_PREFIX, ROOT  # noqa: E402


def main():
    name = sys.argv[1]
    slug = KERNEL_PREFIX + name
    out = os.path.join(ROOT, "runs", name)
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value(api.CONFIG_NAME_USER)

    token, n_new, n_all = None, 0, 0
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = user, slug
            if token:
                request.page_token = token
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files:
                path = os.path.join(out, item.file_name)
                n_all += 1
                if os.path.exists(path):
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(requests.get(item.url).content)
                n_new += 1
            if response.log:
                with open(os.path.join(out, "kaggle.log"), "w", encoding="utf-8") as f:
                    f.write(response.log)
            token = response.next_page_token
            print(f"{n_all} files listed, {n_new} downloaded", flush=True)
            if not token:
                break


if __name__ == "__main__":
    main()
