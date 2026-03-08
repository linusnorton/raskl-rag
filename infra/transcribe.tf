resource "aws_transcribe_vocabulary" "jmbras" {
  vocabulary_name     = "jmbras-proper-names"
  language_code       = "en-GB"
  vocabulary_file_uri = "s3://${aws_s3_bucket.docs.id}/config/transcribe-vocab.csv"

  depends_on = [
    aws_s3_object.transcribe_vocab,
    aws_iam_role_policy.github_actions,
  ]
}

resource "aws_s3_object" "transcribe_vocab" {
  bucket = aws_s3_bucket.docs.id
  key    = "config/transcribe-vocab.csv"
  source = "${path.module}/transcribe-vocab.csv"
  etag   = filemd5("${path.module}/transcribe-vocab.csv")
}
