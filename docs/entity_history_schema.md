# Schema kho lịch sử thực thể V1

## Mục đích

Kho này bổ sung lịch sử cho các thực thể đã xuất hiện trong 1.646 phim modeling. Phim
lịch sử chỉ được dùng để tính feature; chúng không trở thành quan sát có nhãn mới trong
tập huấn luyện hoặc tập đánh giá.

## Snapshot và nguồn

- Schema: `1.0.0`.
- Snapshot: `tmdb-2026-08-05-v1`.
- Nguồn director, cast và production company: TMDb Official API.
- Distributor: bị chặn cho đến khi có nguồn chứng minh được provenance, theatrical role
  và liên kết bằng ID ổn định.
- Không thu thập popularity, vote count, vote average hoặc rating làm predictor.

## File dữ liệu local

Các file dưới `data/external/entity_history/v1/` không được Git theo dõi:

- `movies.jsonl.gz`: một dòng trên mỗi phim lịch sử.
- `directors.jsonl.gz`: quan hệ phim–đạo diễn.
- `production_companies.jsonl.gz`: quan hệ phim–hãng sản xuất.
- `top_cast.jsonl.gz`: quan hệ phim–diễn viên cùng billing order.
- `entities.jsonl.gz`: ID và tên hiển thị của thực thể; tên không phải khóa ghép.
- `manifest.json`: version, nguồn, số dòng, checksum và quy tắc.
- `request_failures.jsonl`: request lỗi không thể phục hồi.

## Trường phim

`movie_tmdb_id`, `imdb_id`, `release_date`, `budget`, `revenue`, các cờ hợp lệ tài
chính, `adult`, `status`, `source`, `source_endpoint`, `retrieved_at_utc`,
`snapshot_version` và `schema_version`.

Budget hoặc revenue không dương được lưu là thiếu. Dữ liệu gốc TMDb không bị sửa.

## Trường quan hệ thực thể

`movie_tmdb_id`, `entity_id`, `entity_type`, `role`, `billing_order`, `credit_id`,
`release_date`, `budget`, `revenue`, các cờ hợp lệ, nguồn, thời điểm thu thập và version.

## Loại trùng

- Phim: `movie_tmdb_id`.
- Director: `(movie_tmdb_id, entity_id, role)`.
- Production company: `(movie_tmdb_id, entity_id)`.
- Cast: ưu tiên `(movie_tmdb_id, credit_id)`; fallback có billing order.
- Không ghép thực thể bằng tên nếu có ID.
- Toàn bộ 1.646 `tmdb_id` mục tiêu bị loại khỏi kho external.

## Production company không phải distributor

`production_companies` mô tả đơn vị tham gia sản xuất. Nó không được đổi tên hoặc dùng
thay cho nhà phân phối. TMDb release dates và watch providers cũng không chứng minh một
đơn vị là theatrical distributor. Nhánh distributor chỉ được mở sau một audit nguồn
riêng và không tạo bản ghi thay thế giả.
