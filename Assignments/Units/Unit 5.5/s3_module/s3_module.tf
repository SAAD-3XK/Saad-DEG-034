resource "aws_s3_bucket" "s3_data" {
    bucket = var.bucket_name
}

resource "aws_s3_object" "folder" {
    bucket = "${aws_s3_bucket.s3_data.id}"
    key    = var.s3_bucket
    source = "/dev/null"
}