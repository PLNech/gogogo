---
layout: default
title: Home
---

<div class="intro">
    <div class="quote">
        一石万局<br>
        <em>One stone, ten thousand games</em>
    </div>

    <p>A journey through AI and Go. Building intelligence through play, making oneself stronger by teaching machines to see.</p>
</div>

<section class="posts-section">
<h2>The Journey</h2>

<ul class="post-list">
{% for post in site.posts reversed %}
{% assign slug = post.slug | default: post.title | slugify %}
<li>
    <div class="post-thumb-container">
        <img class="post-thumb" src="{{ '/assets/hero/' | append: slug | append: '.png' | relative_url }}"
             onerror="this.src='{{ '/assets/hero/' | append: slug | append: '.jpg' | relative_url }}'"
             alt="{{ post.title }}">
    </div>
    <div class="post-content-preview">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time>
        {% if post.excerpt %}
        <p class="excerpt">{{ post.excerpt | strip_html | truncate: 120 }}</p>
        {% endif %}
    </div>
</li>
{% endfor %}
</ul>
</section>
