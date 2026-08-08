# Nguồn và phạm vi sử dụng dữ liệu

## Nguồn dữ liệu chính thức

Báo cáo và mô hình cuối chỉ sử dụng dữ liệu từ **TMDB Official API**. Bộ dữ
liệu phim chính được thu thập ngày 17-07-2026 (UTC), gồm tối đa 100 phim phổ
biến cho mỗi năm trong giai đoạn 2000–2025.

Các trường được lưu từ TMDb gồm mã phim, tiêu đề, ngày phát hành, ngôn ngữ,
ngân sách, doanh thu, thời lượng, thể loại, quốc gia và công ty sản xuất,
collection, trạng thái cùng một số metadata liên quan. Trường `imdb_id` trong
phản hồi TMDb chỉ được xem là mã định danh tham chiếu; mô hình cuối không sử
dụng điểm hoặc số lượt đánh giá từ IMDb.

Các biến phản hồi khán giả của TMDb như `popularity`, `vote_average` và
`vote_count` có thể tồn tại trong dữ liệu gốc nhằm bảo toàn phản hồi API, nhưng
không được sử dụng làm biến dự báo vì không bảo đảm tồn tại trước thời điểm
phát hành.

Nguồn: [The Movie Database (TMDB)](https://www.themoviedb.org/).

> This product uses the TMDB API but is not endorsed or certified by TMDB.

## Chính sách phân phối

Các tệp dữ liệu gốc và dữ liệu dẫn xuất không được phân phối qua repository
công khai. Repository chỉ cung cấp mã nguồn, schema, tài liệu, kết quả tổng hợp
và cấu trúc thư mục. Người dùng muốn tái lập nghiên cứu phải tự thu thập dữ liệu
qua TMDb Official API bằng token của chính mình.

## Artifact ngoài phạm vi cuối

Trong giai đoạn khảo sát ban đầu, dự án từng tải và ghép IMDb Ratings. Mã nguồn
liên quan được giữ để bảo toàn lịch sử kỹ thuật, nhưng các tệp dữ liệu IMDb chỉ
được lưu local và không được phân phối lại. Các giá trị IMDb không tham gia tập
biến dự báo, benchmark XGBoost `0,719483` hoặc báo cáo cuối.
