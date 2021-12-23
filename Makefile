default:
	python3 proj.py

pack: clean
	mkdir -p pack pack/src pack/audio
	cp proj.py pack/src
	cp audio/4cos.wav audio/clean_bandstop.wav audio/xskalo01.wav pack/audio
	cd pack && tar -czvf xskalo01.tar.gz * && cd ..
	rm -rf pack/audio pack/src
	cp doc/doc.pdf pack/xskalo01.pdf

clean:
	rm -rf pack
