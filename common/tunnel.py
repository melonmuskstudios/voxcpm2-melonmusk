"""Make the local server reachable from the internet. Cloudflare quick tunnel by default (no account, no expiry, new link each start);
ngrok when NGROK_TOKEN is set (with NGROK_DOMAIN = your free static domain the link never changes)."""
import os
import re
import subprocess
import threading

CF = "/kaggle/working/cloudflared" if os.path.isdir("/kaggle/working") else "cloudflared"


def start(port, on_url, ngrok_token="", ngrok_domain=""):
    if ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        kw = {"domain": ngrok_domain} if ngrok_domain else {}
        on_url(ngrok.connect(port, **kw).public_url)
        return None
    if not os.path.exists(CF):
        subprocess.run(["wget", "-q", "-O", CF, "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
        os.chmod(CF, 0o755)
    p = subprocess.Popen([CF, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump():
        done = False
        for line in p.stdout:
            m = re.search(r"https://[a-z0-9-]+[.]trycloudflare[.]com", line)
            if m and not done:
                done = True
                on_url(m.group(0))
    threading.Thread(target=pump, daemon=True).start()
    return p
