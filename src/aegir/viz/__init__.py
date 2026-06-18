"""Live HoloViews visualizations served by dedicated `panel serve` processes (topology B).

Served behind the gateway's reverse proxy (single-origin, air-gapped) and embedded into the React app
via ``bokeh.embed.server_document`` — a clean no-iframe DOM embed using the panel server's own BokehJS
(python-bokeh's build, which renders GraphRenderer/Chord correctly; the npm ``@bokeh/bokehjs`` does not).
"""
