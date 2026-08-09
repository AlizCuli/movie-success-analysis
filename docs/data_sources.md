# Nguồn và phạm vi dữ liệu

## Nguồn duy nhất

Báo cáo và mô hình cuối chỉ sử dụng **TMDb Official API**. Ảnh chụp dữ liệu
chính được thu thập ngày 17-07-2026 (UTC), gồm tối đa 100 phim phổ biến cho mỗi
năm trong giai đoạn 2000–2025.

Các endpoint chính:

- `/discover/movie`: xác định danh sách phim theo từng năm;
- `/movie/{movie_id}`: budget, revenue, runtime, genre, quốc gia và công ty;
- `/movie/{movie_id}/credits`, `/release_dates`, `/keywords`: siêu dữ liệu bổ
  sung trong phạm vi trước phát hành theo quy ước vận hành.

Các trường popularity, vote và rating có thể được giữ nguyên trong phản hồi thô
để bảo toàn dữ liệu nguồn, nhưng không được dùng trong EDA chính hoặc predictor
vì không bảo đảm có sẵn trước phát hành. `revenue` chỉ được sử dụng để xác lập
biến mục tiêu.

Nguồn: [The Movie Database (TMDb)](https://www.themoviedb.org/).

> This product uses the TMDB API but is not endorsed or certified by TMDB.

## Chiến lược lấy mẫu

- Giai đoạn: 2000-01-01 đến 2025-12-31.
- Tối đa 5 trang mỗi năm, tương đương tối đa 100 phim mỗi năm.
- Sắp xếp theo `popularity.desc` chỉ phục vụ chọn mẫu khi thu thập;
  `popularity` không phải đặc trưng dự báo.
- Chỉ giữ phim không dành cho người lớn, không phải video và có ngày phát hành
  thực tế trong phạm vi.
- Loại trùng bằng `tmdb_id`.

## Chính sách phân phối

File gốc, trung gian, đã xử lý, điểm kiểm tra và dự đoán cấp từng phim chỉ được
lưu cục bộ và bị `.gitignore` loại khỏi Git. Repository công khai chỉ cung cấp
mã nguồn, cấu trúc thư mục, mô hình đóng gói và kết quả tổng hợp. Việc tái tạo
ảnh chụp dữ liệu yêu cầu TMDb API Read Access Token hợp lệ.

Do TMDb được cập nhật liên tục, số liệu tái chạy vào thời điểm khác có thể thay
đổi nhẹ so với ảnh chụp dữ liệu dùng để thiết lập mốc tham chiếu.
