---
layout: page
comments: true
title:  "Workflow Github Pages"
excerpt: "Some guidance to write blog posts on Github Pages"
date:   2021-05-30 08:53:00
mathjax: true
---

This post gives some guidance on which software too use and how to setup a blog on Github Pages.
The OS I use is Linux (Kubuntu 19.10)




### Install software
#### Jekyll
[jekyll](jekyllrb.com): is a static site generator. You give it text written in your favorite markup language and it uses layouts to create a static website. You can tweak how you want the site URLs to look, what data gets displayed on the site, and more.     

[Instructions](https://jekyllrb.com/docs/):   
- Install Ruby development environment:
  - Install dependencies:
    ```
    sudo apt-get install ruby-full build-essential zlib1g-dev
    ```
  - Add environment variables to your ~/.bashrc.
    ```
    echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
    echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```
- Install Jekyll:
  ```
  gem install jekyll bundler
  ```
- Create a new Jekyll site at ./myblog:
  ```
  jekyll new myblog
  ```
  This automatically creates following files in the ./myblog directory:   
  >
  ./  
  ../  
  _posts/  
  404.html  
  about.markdown  
  _config.yml  
  Gemfile  
  Gemfile.lock  
  .gitignore
  index.markdown

sdafsadf

- Change into your new directory.
  ```
  cd myblog
  ```
- Build the site and make it available on a local server.
  ```
  bundle exec jekyll serve
  ```






https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet


### Setup Github



### Configure ...
