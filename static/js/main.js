$(document).ready(function() {
    // Variables
    let uploadedFile = null;
    let filePath = null;
    
    // Basic elements
    const fileInput = $('#fileInput');
    const previewContainer = $('#previewContainer');
    const previewImage = $('#previewImage');
    const searchButton = $('#searchButton');
    const removePreview = $('#removePreview');
    const selectImageBtn = $('#selectImageBtn');
    
    // Handle click on select image button
    selectImageBtn.on('click', function(e) {
        e.preventDefault();
        fileInput.click();
        console.log("Select image button clicked");
    });
    
    // Handle file selection
    fileInput.on('change', function(e) {
        const files = this.files;
        if (files.length > 0) {
            console.log("File selected via input: ", files[0].name);
            handleFile(files[0]);
        }
    });
    
    // Remove preview
    removePreview.on('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        clearPreview();
        console.log("Preview removed");
    });
    
    // Search button click
    searchButton.on('click', searchSimilarFaces);
    
    // Database management buttons
    $('#buildDatabaseButton').on('click', buildDatabase);
    $('#clearDatabaseButton').on('click', clearDatabase);
    
    function handleFile(file) {
        console.log("Handling file: ", file.name, file.type, file.size);
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            alert('Vui lòng chọn file ảnh (jpg, jpeg, png, hoặc bmp)');
            return;
        }
        
        // Validate file size (10MB max)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            alert('Kích thước file quá lớn. Vui lòng chọn file nhỏ hơn 10MB.');
            return;
        }
        
        // Store file for upload
        uploadedFile = file;
        
        // Preview image
        try {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.attr('src', e.target.result);
                previewContainer.show();
                searchButton.prop('disabled', false);
                console.log("Image preview displayed");
            };
            
            reader.onerror = function(e) {
                console.error("FileReader error: ", e);
                alert("Lỗi khi đọc file. Vui lòng thử lại.");
            };
            
            reader.readAsDataURL(file);
        } catch (error) {
            console.error("Exception when reading file: ", error);
            alert("Lỗi khi đọc file: " + error.message);
        }
    }
    
    function clearPreview() {
        uploadedFile = null;
        filePath = null;
        previewImage.attr('src', '');
        previewContainer.hide();
        searchButton.prop('disabled', true);
    }
    
    function searchSimilarFaces() {
        console.log("=== Search function triggered ===");
        
        if (!uploadedFile && !filePath) {
            alert('Vui lòng tải lên ảnh trước khi tìm kiếm.');
            console.error("No file uploaded");
            return;
        }
        
        // Show loading
        $('#searchResults').hide();
        $('#loading').show();
        console.log("Loading indicator shown");
        
        // If already have a file path (file was uploaded previously)
        if (filePath) {
            console.log("Using existing file path:", filePath);
            performSearch(filePath);
            return;
        }
        
        // Upload file first
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        console.log("Uploading file for search:", uploadedFile.name, "size:", uploadedFile.size, "type:", uploadedFile.type);
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            timeout: 30000, // 30 seconds timeout
            success: function(response) {
                console.log("Upload response:", response);
                if (response.success) {
                    filePath = response.file_path;
                    console.log("File uploaded successfully, path:", filePath);
                    performSearch(filePath);
                } else {
                    console.error("Upload failed:", response.error);
                    handleError(response.error || 'Lỗi không xác định khi tải lên ảnh');
                }
            },
            error: function(xhr, status, error) {
                console.error('Upload error:', xhr.status, error);
                console.error('Response text:', xhr.responseText);
                if (status === 'timeout') {
                    handleError('Quá thời gian tải lên. Vui lòng thử lại.');
                } else {
                    handleError('Lỗi khi tải lên: ' + error);
                }
            }
        });
    }
    
    function performSearch(imagePath) {
        console.log("=== Performing search ===");
        console.log("Image path for search:", imagePath);
        
        const formData = new FormData();
        formData.append('file_path', imagePath);
        
        console.log("Form data prepared for search, keys:", [...formData.keys()]);
        console.log("Form data values:", [...formData.values()]);
        
        $.ajax({
            url: '/search',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            timeout: 60000, // 60 seconds timeout for search
            success: function(response) {
                console.log("Search response received:", response);
                $('#loading').hide();
                
                if (response.success) {
                    console.log("Search successful, results:", response.results ? response.results.length : 0);
                    displaySearchResults(response);
                } else {
                    console.error("Search failed:", response.error);
                    handleError(response.error || 'Lỗi không xác định khi tìm kiếm');
                }
            },
            error: function(xhr, status, error) {
                console.error('Search error status:', status);
                console.error('Search error code:', xhr.status);
                console.error('Search error message:', error);
                
                try {
                    if (xhr.responseText) {
                        console.error('Response text:', xhr.responseText);
                        const errorResponse = JSON.parse(xhr.responseText);
                        if (errorResponse && errorResponse.error) {
                            handleError('Lỗi khi tìm kiếm: ' + errorResponse.error);
                            return;
                        }
                    }
                } catch (e) {
                    console.error('Error parsing error response:', e);
                }
                
                if (status === 'timeout') {
                    handleError('Quá thời gian tìm kiếm. Vui lòng thử lại.');
                } else {
                    handleError('Lỗi khi tìm kiếm: ' + error);
                }
                
                $('#loading').hide();
            }
        });
    }
    
    function displaySearchResults(data) {
        console.log("=== Displaying search results ===");
        const resultsContainer = $('#searchResults');
        
        try {
            console.log("Clearing results container");
            resultsContainer.empty();
            
            if (!data || typeof data !== 'object') {
                console.error("Invalid data format:", data);
                throw new Error("Lỗi: Dữ liệu không hợp lệ");
            }
            
            // Show query image
            if (!data.query) {
                console.error("No query data in response");
                throw new Error("Lỗi: Thiếu thông tin ảnh truy vấn");
            }
            
            console.log("Query data:", data.query);
            const query = data.query;
            
            // Fix any malformed image paths
            const queryImagePath = query.image_path ? query.image_path.replace(/\\/g, '/') : '/static/images/error.png';
            console.log("Query image path:", queryImagePath);
            
            const queryHtml = `
                <div class="mb-4">
                    <h4>Ảnh tìm kiếm</h4>
                    <div class="card">
                        <img src="${queryImagePath}" class="card-img-top" alt="Query Image" onerror="this.onerror=null; this.src='/static/images/error.png'; console.error('Failed to load query image:', this.src);">
                        <div class="card-body">
                            <div>
                                <span class="badge badge-primary">Giới tính: ${query.gender || 'Không xác định'}</span>
                                <span class="badge badge-success">Màu da: ${query.skin_color || 'Không xác định'}</span>
                                <span class="badge badge-info">Cảm xúc: ${query.emotion || 'Không xác định'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            console.log("Query HTML prepared");
            resultsContainer.append(queryHtml);
            console.log("Query image appended to container");
            
            // Show similar faces
            if (!data.results || !Array.isArray(data.results)) {
                console.error("No valid results data in response");
                resultsContainer.append('<div class="alert alert-warning">Không tìm thấy kết quả hoặc dữ liệu không hợp lệ.</div>');
                resultsContainer.show();
                return;
            }
            
            const results = data.results;
            console.log("Results data:", results);
            
            if (results && results.length > 0) {
                console.log("Displaying", results.length, "similar faces");
                resultsContainer.append('<h4>Kết quả tìm kiếm</h4>');
                
                const resultsHtml = $('<div class="row"></div>');
                results.forEach((result, index) => {
                    console.log("Processing result #", index + 1, ":", result);
                    
                    // Validate result data
                    if (!result || typeof result !== 'object') {
                        console.error("Invalid result format:", result);
                        return; // Skip this result
                    }
                    
                    // Create result card
                    try {
                        const resultRank = result.rank ? 
                            (typeof result.rank === 'string' ? result.rank.replace('Top ', '') : result.rank) : 
                            (index + 1);
                        
                        // Fix any malformed image paths
                        const resultImagePath = result.image_path ? result.image_path.replace(/\\/g, '/') : '/static/images/error.png';
                        console.log("Result image path:", resultImagePath);
                        
                        const resultCard = `
                            <div class="col-md-4 mb-4">
                                <div class="card h-100">
                                    <img src="${resultImagePath}" class="card-img-top" alt="Similar Face" onerror="this.onerror=null; this.src='/static/images/error.png'; console.error('Failed to load result image:', this.src);">
                                    <div class="card-body">
                                        <h5 class="card-title">Kết quả #${resultRank}</h5>
                                        <div>
                                            <span class="badge badge-primary">Giới tính: ${result.gender || 'Không xác định'}</span>
                                            <span class="badge badge-success">Màu da: ${result.skin_color || 'Không xác định'}</span>
                                            <span class="badge badge-info">Cảm xúc: ${result.emotion || 'Không xác định'}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                        resultsHtml.append(resultCard);
                    } catch (err) {
                        console.error("Error creating result card:", err, "for result:", result);
                    }
                });
                
                console.log("Results HTML prepared");
                resultsContainer.append(resultsHtml);
                console.log("Results appended to container");
            } else {
                console.log("No similar faces found");
                resultsContainer.append('<div class="alert alert-info">Không tìm thấy khuôn mặt tương tự.</div>');
            }
            
            console.log("Showing results container");
            resultsContainer.show();
        } catch (error) {
            console.error("Error displaying results:", error);
            resultsContainer.html(`
                <div class="alert alert-danger">
                    <strong>Lỗi!</strong> ${error.message || "Lỗi hiển thị kết quả"}
                </div>
            `);
            resultsContainer.show();
        }
    }
    
    function buildDatabase() {
        const folderPath = $('#folderPath').val().trim();
        
        if (!folderPath) {
            alert('Vui lòng nhập đường dẫn thư mục.');
            return;
        }
        
        // Show loading
        $('#databaseMessage').hide();
        $('#databaseLoading').show();
        
        console.log("Building database from folder:", folderPath);
        
        $.ajax({
            url: '/build-database',
            type: 'POST',
            data: JSON.stringify({ folder_path: folderPath }),
            contentType: 'application/json',
            success: function(response) {
                console.log("Build database response:", response);
                $('#databaseLoading').hide();
                
                if (response.success) {
                    showDatabaseMessage('success', response.message);
                } else {
                    showDatabaseMessage('danger', response.error || 'Lỗi không xác định khi xây dựng CSDL');
                }
            },
            error: function(xhr, status, error) {
                console.error('Database build error:', xhr.responseText);
                $('#databaseLoading').hide();
                showDatabaseMessage('danger', 'Lỗi khi xây dựng CSDL: ' + error);
            }
        });
    }
    
    function clearDatabase() {
        if (!confirm('Bạn có chắc chắn muốn xóa toàn bộ cơ sở dữ liệu không?')) {
            return;
        }
        
        // Show loading
        $('#databaseMessage').hide();
        $('#databaseLoading').show();
        
        console.log("Clearing database");
        
        $.ajax({
            url: '/clear-database',
            type: 'POST',
            success: function(response) {
                console.log("Clear database response:", response);
                $('#databaseLoading').hide();
                
                if (response.success) {
                    showDatabaseMessage('success', response.message);
                } else {
                    showDatabaseMessage('danger', response.error || 'Lỗi không xác định khi xóa CSDL');
                }
            },
            error: function(xhr, status, error) {
                console.error('Database clear error:', xhr.responseText);
                $('#databaseLoading').hide();
                showDatabaseMessage('danger', 'Lỗi khi xóa CSDL: ' + error);
            }
        });
    }
    
    function showDatabaseMessage(type, message) {
        const messageDiv = $('#databaseMessage');
        messageDiv.removeClass('alert-success alert-danger').addClass('alert-' + type);
        messageDiv.text(message);
        messageDiv.show();
    }
    
    function handleError(message) {
        $('#loading').hide();
        $('#searchResults').html(`
            <div class="alert alert-danger">
                <strong>Lỗi!</strong> ${message}
            </div>
        `);
        $('#searchResults').show();
    }
}); 