# Phát hiện EDA về dữ liệu phim

> **Phạm vi:** Các phân tích rating, vote và popularity chỉ có vai trò mô tả
> và minh họa rủi ro leakage. Mô hình XGBoost cuối không sử dụng các biến này.
> EDA không tham gia lựa chọn đặc trưng hoặc tinh chỉnh mô hình bằng kết quả
> kiểm định outer.

## Phát hiện chính

1. Tập `movies_cleaned` có 2.597 phim và giữ toàn bộ quan sát; tập
   `movies_modeling` có 1.646 phim đủ điều kiện về budget, revenue, runtime,
   ngày phát hành và nhãn. Trong tập làm sạch, budget thiếu 855 phim (32,92%),
   revenue thiếu 839 phim (32,31%) và runtime thiếu 72 phim (2,77%).
2. Budget và revenue phân phối lệch phải: skewness lần lượt là 1,75 và 2,89.
   Median budget là 48,0 triệu USD, thấp hơn mean 69,2 triệu USD; median revenue
   là 139,7 triệu USD, thấp hơn mean 240,7 triệu USD. Vì vậy các biểu đồ và
   phân tích quan hệ tài chính dùng thêm thang `log1p`.
3. Trong tập modeling, revenue có tương quan Pearson/Spearman mạnh với budget
   (0,670/0,745). Pearson và Spearman khác nhau vì Pearson đo quan hệ tuyến tính
   và nhạy với ngoại lai, còn Spearman đo quan hệ đơn điệu theo thứ hạng.
4. Action là thể loại xuất hiện nhiều nhất trong phân tích đa thể loại với 623
   phim; Adventure có median revenue 334,0 triệu USD. Trong các thể loại có ít
   nhất 30 phim, Romance có tỷ lệ thành công cao nhất 77,94%, còn History thấp
   nhất 54,24%. Một phim có thể góp mặt ở nhiều thể loại trong phân tích này.
5. Tỷ lệ thành công theo năm biến động; ví dụ năm 2020 là 40,91%, thấp hơn mức
   66,67% của năm 2025. Theo tháng phát hành, tháng 1 cao nhất (75,68%) và
   tháng 10 thấp nhất (64,12%). Giá trị tiền chưa điều chỉnh lạm phát và mỗi năm
   chỉ có tối đa 100 phim phổ biến theo chiến lược lấy mẫu, nên các đường xu
   hướng chỉ mô tả mẫu hiện có, không đại diện cho toàn thị trường.
6. Nhãn `is_successful` mất cân bằng vừa phải: 1.169 phim thành công (71,02%) và
   477 phim không thành công (28,98%). Median revenue của hai lớp là 206,4 và
   55,3 triệu USD; median runtime là 112 và 109 phút. Khác biệt revenue là hệ
   quả trực tiếp một phần của định nghĩa nhãn, nên không dùng nó làm biến dự
   đoán.
7. Kiểm định Chi-square giữa `primary_genre` (đã gộp thể loại dưới 30 phim vào
   `Other`) và `is_successful` cho χ²(11) = 24,626, p = 0,0103. Điều kiện tần số
   kỳ vọng đạt (nhỏ nhất 10,72; không ô nào dưới 5), nhưng Cramér’s V = 0,1223
   cho thấy mức liên hệ yếu. Vì vậy, ý nghĩa thống kê không đồng nghĩa với tác
   động thực tế mạnh.

## Dữ liệu thiếu và ngoại lai

- Budget và revenue bằng 0 đã được coi là chưa công bố và chuyển thành thiếu ở
  bước tiền xử lý; không điền giá trị thiếu trong EDA.
- Có 50 budget và 135 revenue được đánh dấu ngoại lai bằng IQR. Các quan sát này
  vẫn được giữ vì có thể là phim bom tấn hoặc trường hợp có ý nghĩa.
- Tỷ lệ `revenue_to_budget` rất lệch (skewness 40,55), vì vậy median và thang
  log là cách mô tả ổn định hơn khi phù hợp.

## Hạn chế diễn giải

- Tương quan không chứng minh quan hệ nhân quả; các yếu tố như marketing, thị
  trường phát hành và thời điểm công bố dữ liệu có thể là biến gây nhiễu.
- Dữ liệu là mẫu tối đa 100 phim phổ biến mỗi năm từ TMDb, không phải mẫu ngẫu
  nhiên của toàn bộ phim điện ảnh.
- Budget và revenue là giá trị danh nghĩa, chưa điều chỉnh lạm phát.
- `popularity`, vote count và rating có thể được cập nhật sau phát hành nên bị
  loại khỏi mô hình và nội dung phân tích chính của báo cáo cuối.

## Hàm ý đối với mô hình

- Các biến nên xem xét trước tiên: `log_budget`, `runtime`, `release_year`,
  `release_month`, `original_language`, thể loại, quốc gia/công ty sản xuất và
  các cờ dữ liệu sẵn có.
- Không dùng `revenue`, `profit`, `roi` hoặc `revenue_to_budget` làm biến đầu
  vào. `is_successful` chỉ được sử dụng làm biến mục tiêu của bài toán phân loại.
