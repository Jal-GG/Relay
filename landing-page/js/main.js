// Copy to clipboard functionality
function copyCode(btn) {
    const code = btn.getAttribute('data-code');
    navigator.clipboard.writeText(code).then(() => {
        const originalText = btn.innerText;
        btn.innerText = 'COPIED!';
        btn.style.color = 'var(--accent-neon)';
        
        setTimeout(() => {
            btn.innerText = originalText;
            btn.style.color = 'var(--text-muted)';
        }, 2000);
    });
}

// Topology Animation
document.addEventListener('DOMContentLoaded', () => {
    const svg = document.getElementById('connections-svg');
    const master = document.getElementById('node-master');
    const cloud = document.getElementById('node-cloud');
    const localCpu = document.getElementById('node-local-cpu');
    const localGpu = document.getElementById('node-local-gpu');
    const mobile = document.getElementById('node-mobile');
    const mini = document.getElementById('node-mini');
    
    if (!svg || !master || !cloud) return;

    let packetInterval;

    function drawConnections() {
        // Clear previous lines
        svg.innerHTML = '';
        
        const containerRect = document.getElementById('topology').getBoundingClientRect();
        
        // Helper to get center of the ICON, not the entire node container,
        // so lines don't cross into the text
        const getCenter = (nodeElement) => {
            const iconElement = nodeElement.querySelector('.node-icon');
            const targetElement = iconElement || nodeElement;
            const rect = targetElement.getBoundingClientRect();
            
            return {
                x: rect.left - containerRect.left + rect.width / 2,
                y: rect.top - containerRect.top + rect.height / 2
            };
        };
        
        const nodes = {
            master: getCenter(master),
            cloud: getCenter(cloud),
            localCpu: getCenter(localCpu),
            localGpu: getCenter(localGpu),
            mobile: getCenter(mobile),
            mini: getCenter(mini)
        };
        
        // Master connects to all
        drawLine(nodes.master, nodes.cloud, 'conn-m-cloud');
        drawLine(nodes.master, nodes.localCpu, 'conn-m-lcpu');
        drawLine(nodes.master, nodes.localGpu, 'conn-m-lgpu');
        drawLine(nodes.master, nodes.mobile, 'conn-m-mobile');
        drawLine(nodes.master, nodes.mini, 'conn-m-mini');

        // Circle/Pentagon mesh perimeter connections
        drawLine(nodes.cloud, nodes.localCpu, 'conn-c-lcpu');
        drawLine(nodes.localCpu, nodes.localGpu, 'conn-lcpu-lgpu');
        drawLine(nodes.localGpu, nodes.mobile, 'conn-lgpu-mob');
        drawLine(nodes.mobile, nodes.mini, 'conn-mob-mini');
        drawLine(nodes.mini, nodes.cloud, 'conn-mini-c');
    }
    
    function drawLine(start, end, id) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const d = `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
        line.setAttribute('d', d);
        line.setAttribute('class', 'conn-line');
        line.setAttribute('id', id);
        svg.appendChild(line);
    }
    
    // Initial draw
    // Use a slight timeout to ensure fonts and layout have fully painted before calculating rects
    setTimeout(drawConnections, 100);
    
    // Redraw on resize
    window.addEventListener('resize', drawConnections);
    
    // Animation sequence
    function playAnimation() {
        // Reset states
        const nodes = [master, cloud, localCpu, localGpu, mobile, mini];
        nodes.forEach(n => n.classList.remove('active'));
        
        const lines = document.querySelectorAll('.conn-line');
        lines.forEach(l => l.classList.remove('active'));
        
        // Sequence: Master -> Cloud -> Local CPU -> Local GPU -> Mobile -> Mini PC -> Master
        function stage(fromNode, toNode, lineId, nextCallback) {
            fromNode.classList.add('active');
            const line = document.getElementById(lineId);
            if (line) line.classList.add('active');
            
            spawnPacket(fromNode, toNode, () => {
                fromNode.classList.remove('active');
                if (line) line.classList.remove('active');
                if (nextCallback) nextCallback();
            });
        }
        
        // Start Pipeline
        stage(master, cloud, 'conn-m-cloud', () => {
            stage(cloud, localCpu, 'conn-c-lcpu', () => {
                stage(localCpu, localGpu, 'conn-lcpu-lgpu', () => {
                    stage(localGpu, mobile, 'conn-lgpu-mob', () => {
                        stage(mobile, mini, 'conn-mob-mini', () => {
                            stage(mini, master, 'conn-m-mini', () => {
                                master.classList.add('active');
                                setTimeout(() => master.classList.remove('active'), 500);
                            });
                        });
                    });
                });
            });
        });
    }
    
    function spawnPacket(startNode, endNode, callback) {
        const packet = document.createElement('div');
        packet.className = 'packet';
        document.getElementById('topology').appendChild(packet);
        
        const containerRect = document.getElementById('topology').getBoundingClientRect();
        
        // Calculate spawn point relative to icon, not container
        const startIcon = startNode.querySelector('.node-icon') || startNode;
        const endIcon = endNode.querySelector('.node-icon') || endNode;
        
        const startRect = startIcon.getBoundingClientRect();
        const endRect = endIcon.getBoundingClientRect();
        
        const startX = startRect.left - containerRect.left + startRect.width / 2;
        const startY = startRect.top - containerRect.top + startRect.height / 2;
        
        const endX = endRect.left - containerRect.left + endRect.width / 2;
        const endY = endRect.top - containerRect.top + endRect.height / 2;
        
        packet.style.left = startX + 'px';
        packet.style.top = startY + 'px';
        packet.style.opacity = '1';
        
        const duration = 600; // ms
        const startTime = performance.now();
        
        function step(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const easeProgress = progress < 0.5 ? 2 * progress * progress : 1 - Math.pow(-2 * progress + 2, 2) / 2;
            
            const currentX = startX + (endX - startX) * easeProgress;
            const currentY = startY + (endY - startY) * easeProgress;
            
            packet.style.left = currentX + 'px';
            packet.style.top = currentY + 'px';
            
            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                packet.style.opacity = '0';
                setTimeout(() => packet.remove(), 200);
                if (callback) callback();
            }
        }
        
        requestAnimationFrame(step);
    }
    
    // Start loop
    setTimeout(() => {
        playAnimation();
        packetInterval = setInterval(playAnimation, 5000); 
    }, 600);
});

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(9, 9, 11, 0.95)';
        navbar.style.boxShadow = '0 4px 30px rgba(0, 0, 0, 0.5)';
    } else {
        navbar.style.background = 'rgba(9, 9, 11, 0.8)';
        navbar.style.boxShadow = 'none';
    }
});
