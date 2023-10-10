---
layout: post
title:  "Jekyll Installation on macOS Catalina"
date:   2021-03-22 22:43:12 -0400
tags: jekyll macOS
categories: coding
---
Due to new security features in macOS Mojave and later, the [tutorial on Jekyll website](https://jekyllrb.com/docs/installation/macos/) will raise an error when running command: `gem install <PACKAGE>`:

```console
ERROR:  Loading command: install (LoadError)
    cannot load such file -- openssl
ERROR:  While executing gem ... (NoMethodError)
    undefined method `invoke_with_build_args' for nil:NilClass
```

The solution is to reinstall ruby with openssl, as discussed in [this thread](https://github.com/rvm/rvm/issues/4819):

* Step 1: install openssl: `brew install rbenv/tap/openssl@1.0`
* Step 2: install/reinstall ruby with openssl: `rvm install <RUBY_VERSION> --with-openssl-dir='/usr/local/opt/openssl@1.0'` (if reinstall, replace `install` with `reinstall`)
* Step 3: update gem: `gem update --system`

You are good to go! Have fun with gem!