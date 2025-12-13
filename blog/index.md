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
<li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time>
    {% if post.excerpt %}
    <p class="excerpt">{{ post.excerpt | strip_html | truncate: 120 }}</p>
    {% endif %}
</li>
{% endfor %}
</ul>
</section>
