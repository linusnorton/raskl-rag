# --- SES email identity for diff notifications ---

resource "aws_ses_email_identity" "notify" {
  email = "linusnorton@gmail.com"
}
