from django import forms

class UrlForm(forms.Form):
  url = forms.URLField(label='推定する記事のURL')
