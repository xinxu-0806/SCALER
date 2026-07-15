all:
	(cd lib; $(MAKE) CFLAGS="$(CFLAGS)")
	(cd util; $(MAKE) CFLAGS="$(CFLAGS)")

lib:
	(cd lib; $(MAKE) CFLAGS="$(CFLAGS)")

util:
	(cd util; $(MAKE) CFLAGS="$(CFLAGS)")

demo:
	(cd demo; $(MAKE) CFLAGS="$(CFLAGS)")

clean:
	(cd lib; $(MAKE) clean)
	(cd util; $(MAKE) clean)
	(cd demo; $(MAKE) clean)
