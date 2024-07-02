#!/bin/bash

export TWEAK_VERSION=$1
export PKG_SERVER_HOST=0.0.0.0
export PKG_SERVER_PORT=8099


pip -v -q install --extra-index-url http://${PKG_SERVER_HOST}:${PKG_SERVER_PORT}/simple --trusted-host=${PKG_SERVER_HOST} tweak==${TWEAK_VERSION}

echo "tweak-${TWEAK_VERSION} installation completed!"
