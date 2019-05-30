from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Markup
app = Flask(__name__)

from app import views
