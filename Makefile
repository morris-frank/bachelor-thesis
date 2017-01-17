PYTHON35 := $(shell which python3.5 2> /dev/null)

all:
	ifnndef PYTHON35
		$(error "Python 3.5 is not installed on the system. Please do this.")
	endif

install:
	[ ! -d .venv ] && virtualenv --system-site-packages --python=python3.5 --prompt=BA .venv; \
	[ ! -d lib ] && mkdir lib; \
	[ ! -d lib/caffe-tensorflow ] && git clone https://github.com/ethereon/caffe-tensorflow lib/caffe-tensorflow; \
	source .venv/bin/activate; \
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0a0-cp35-cp35m-linux_x86_64.whl	
	pip install -r requirements.txt; \

clean:
	rm -fr .venv; \
	rm -fr lib; \
