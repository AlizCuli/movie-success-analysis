# Phạm vi trước phát hành theo quy ước vận hành

## Quy ước phạm vi

Dự án sử dụng các trường TMDb có ý nghĩa trước phát hành: ngân sách, thời lượng,
lịch phát hành, thể loại, ngôn ngữ, quốc gia, công ty sản xuất, collection,
phân loại độ tuổi, quy mô diễn viên/đoàn làm phim, thông tin phát hành, mô tả,
tagline và từ khóa. Phạm vi này được gọi là **trước phát hành theo quy ước vận
hành** vì ảnh chụp TMDb được tải ở hiện tại, không phải kho lưu trữ chứng minh
thời điểm công bố của từng trường.

`revenue` chỉ được sử dụng để tạo biến mục tiêu. `popularity`, điểm đánh giá,
lượt bình chọn, `profit`, ROI và mọi biến dẫn xuất từ doanh thu không được sử
dụng làm đặc trưng dự báo.

## Tập đặc trưng chính thức

- **Siêu dữ liệu cơ bản:** `log_budget`, thời lượng, năm/tháng/mùa phát hành,
  thể loại đa nhãn, ngôn ngữ, quốc gia và công ty sản xuất.
- **Siêu dữ liệu bổ sung:** collection, số công ty/ngôn ngữ, quy mô diễn viên và
  đoàn làm phim, phân loại độ tuổi, số quốc gia/sự kiện chiếu rạp, độ dài mô tả,
  tagline và số từ khóa.
- **Lịch sử franchise theo thời điểm:** số phim trước đó, tỷ lệ thành công lịch
  sử có làm trơn, trung bình `log_budget` lịch sử và số năm kể từ phần phim gần
  nhất.

## Giao thức chống rò rỉ dữ liệu

1. Giữ cố định 1.646 phim, biến mục tiêu và `StratifiedKFold` 5 vòng ngoài với
   seed 42.
2. `StratifiedKFold` 4 vòng trong với seed 43 được sử dụng để chọn số vòng
   boosting và ngưỡng phân loại.
3. Bộ điền khuyết, bộ mã hóa, từ vựng thể loại và bộ kiến tạo lịch sử chỉ được
   khớp trên phân hoạch huấn luyện tương ứng.
4. Lịch sử franchise chỉ sử dụng phim có `release_date` sớm hơn phim truy vấn;
   các phim cùng ngày không tạo lịch sử cho nhau.
5. Phân hoạch vòng ngoài chỉ phục vụ đánh giá cuối cùng và không được sử dụng
   lại để lựa chọn đặc trưng hoặc tham số.

## Mốc tham chiếu

XGBoost A+B kết hợp lịch sử franchise đạt Macro-F1 ngoài mẫu gộp **0,719483**, F1
lớp 0 **0,605128**, recall lớp 0 **0,618449** và balanced accuracy **0,722398**.
